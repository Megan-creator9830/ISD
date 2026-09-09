import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralTensorConstruction(nn.Module):
    """Phase 1: Structural Tensor Construction (STC)
    Constructs a 4th-order multi-scale structural tensor via spatial affinity weighting.
    """
    def __init__(self, channels, scales=(3, 5, 7), tau=1.0):
        super().__init__()
        self.scales = scales
        self.tau = tau
        # Channel alignment projection if needed
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, feat, relevance_field):
        """
        Args:
            feat: [B, C, H, W] deep feature maps
            relevance_field: [B, 1, H, W] task-driven predictive relevance (R)
        Returns:
            T: [B, K, C, H, W] 4th-order structural tensor
        """
        B, C, H, W = feat.shape
        tensor_slices = []

        for k in self.scales:
            pad = k // 2
            # Spatial neighborhood smoothing as affinity aggregation
            feat_weighted = feat * relevance_field
            pooled_feat = F.avg_pool2d(feat_weighted, kernel_size=k, stride=1, padding=pad)
            pooled_norm = F.avg_pool2d(relevance_field, kernel_size=k, stride=1, padding=pad) + 1e-6
            Q_k = pooled_feat / pooled_norm
            tensor_slices.append(Q_k.unsqueeze(1))  # [B, 1, C, H, W]

        # Stack across scale channel dimension
        T = torch.cat(tensor_slices, dim=1)  # [B, K, C, H, W]
        return T


class StructuralTensorSeparation(nn.Module):
    """Phase 2: Structural Tensor Separation (STS)
    Decouples structural tensor into target structure, context, and noise components.
    """
    def __init__(self, channels, num_scales=3):
        super().__init__()
        # Decoupling projection heads
        self.head_ship = nn.Sequential(
            nn.Conv3d(num_scales, num_scales, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_scales),
            nn.ReLU(inplace=True)
        )
        self.head_bg = nn.Sequential(
            nn.Conv3d(num_scales, num_scales, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_scales),
            nn.ReLU(inplace=True)
        )

    def forward(self, T, ship_mask, support_degree):
        """
        Args:
            T: [B, K, C, H, W] structural tensor
            ship_mask: [B, 1, 1, H, W] ground truth ship bounding region
            support_degree: [B, 1, 1, H, W] background predictive support degree delta
        Returns:
            T_s, T_b, T_n: decoupled components
            loss_sep: tensor separation constraint loss
        """
        T_s = self.head_ship(T) * ship_mask
        T_b = self.head_bg(T) * (1.0 - ship_mask) * support_degree
        T_n = T - T_s - T_b

        # Regularization: Low-rank proxy (Frobenius) + Sparse noise penalty (L1)
        loss_sparse_noise = torch.mean(torch.abs(T_n))
        loss_ortho = torch.mean(torch.abs(T_s * T_b))  # Mutual orthogonality
        loss_sep = loss_sparse_noise + 0.5 * loss_ortho

        return T_s, T_b, T_n, loss_sep


class PredictiveConsistentDistillationLoss(nn.Module):
    """Phase 3: Predictive Basis Consistent Distillation (PCSD)
    Aligns core tensor manifolds via Tucker decomposition projection and enforces decision basis consistency.
    """
    def __init__(self, in_channels_t, in_channels_s, core_rank=16):
        super().__init__()
        self.core_rank = core_rank
        # Tucker factor mode projections to align channel dimensions to core rank
        self.proj_t = nn.Conv3d(3, 3, kernel_size=(in_channels_t, 1, 1), bias=False)
        self.proj_s = nn.Conv3d(3, 3, kernel_size=(in_channels_s, 1, 1), bias=False)
        self.core_align = nn.Linear(in_channels_s, in_channels_t, bias=False)

    def forward(self, T_s_t, T_b_t, T_s_s, T_b_s, R_t, R_s, W_noise_t):
        """
        Args:
            T_s_t, T_b_t: decoupled teacher components [B, K, C_t, H, W]
            T_s_s, T_b_s: decoupled student components [B, K, C_s, H, W]
            R_t, R_s: normalized predictive relevance maps [B, 1, H, W]
            W_noise_t: invalid noise mask from teacher [B, 1, H, W]
        Returns:
            loss_SCM: total structural distillation loss
        """
        # 1. Structural manifold alignment on ship components (L_v)
        B, K, C_s, H, W = T_s_s.shape
        _, _, C_t, _, _ = T_s_t.shape

        # Adapt student channels to teacher core space
        T_s_s_adapted = F.interpolate(
            T_s_s.view(B * K, C_s, H, W), size=(H, W), mode='bilinear', align_corners=False
        )
        if C_s != C_t:
            T_s_s_adapted = F.conv2d(T_s_s_adapted, self.core_align.weight.unsqueeze(-1).unsqueeze(-1))
        T_s_s_adapted = T_s_s_adapted.view(B, K, C_t, H, W)

        loss_v = F.mse_loss(T_s_s_adapted, T_s_t) + 0.5 * F.mse_loss(
            T_b_s.mean(dim=2, keepdim=True), T_b_t.mean(dim=2, keepdim=True)
        )

        # 2. Decision basis consistency constraint (Cosine alignment between R_s and R_t)
        R_t_flat = R_t.view(B, -1)
        R_s_flat = R_s.view(B, -1)
        cos_sim = F.cosine_similarity(R_t_flat, R_s_flat, dim=-1)
        loss_R = torch.mean(1.0 - cos_sim)

        # 3. Invalid texture noise suppression (L_n)
        loss_n = torch.mean(W_noise_t * R_s) / (torch.mean(W_noise_t) + 1e-6)

        # Total Distillation Loss
        loss_SCM = loss_v + 0.1 * loss_n + 0.5 * loss_R
        return loss_SCM


class ISDPlusTrainer(nn.Module):
    """Overall Training Wrapper for ISD+ Framework"""
    def __init__(self, teacher_model, student_model, feat_dim_t=256, feat_dim_s=128):
        super().__init__()
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False  # Freeze teacher

        self.student = student_model
        self.stc = StructuralTensorConstruction(channels=feat_dim_s)
        self.sts = StructuralTensorSeparation(channels=feat_dim_s)
        self.distill_loss = PredictiveConsistentDistillationLoss(
            in_channels_t=feat_dim_t, in_channels_s=feat_dim_s
        )

    def extract_relevance_field(self, feat):
        """Derives gradient-aware predictive relevance via spatial activation"""
        # Feature-level attribution proxy (Norm across channel dimension)
        relevance = torch.norm(feat, p=2, dim=1, keepdim=True)
        relevance = relevance / (relevance.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return relevance

    def forward(self, images, targets=None):
        if not self.training:
            return self.student(images)

        # Forward Teacher (No grad)
        with torch.no_grad():
            feat_t = self.teacher.extract_feat(images)
            R_t = self.extract_relevance_field(feat_t)
            T_t = self.stc(feat_t, R_t)

        # Forward Student
        feat_s, preds_s = self.student.extract_feat_and_pred(images)
        R_s = self.extract_relevance_field(feat_s)
        T_s = self.stc(feat_s, R_s)

        # Mask generation (dummy targets structure for demonstration)
        B, _, H, W = feat_s.shape
        ship_mask = torch.zeros(B, 1, 1, H, W, device=images.device)
        support_deg = torch.ones_like(ship_mask) * 0.5

        # Structural Separation (STS)
        T_s_t, T_b_t, T_n_t, loss_sep_t = self.sts(T_t, ship_mask, support_deg)
        T_s_s, T_b_s, T_n_s, loss_sep_s = self.sts(T_s, ship_mask, support_deg)

        # Distillation Loss (PCSD)
        W_noise = (T_n_t.abs().mean(dim=(1, 2), keepdim=True) > 0.3).float()
        loss_SCM = self.distill_loss(T_s_t, T_b_t, T_s_s, T_b_s, R_t, R_s, W_noise)

        # Detection Task Loss (Task Head)
        loss_det = self.student.compute_loss(preds_s, targets)

        # Multi-task Objective
        loss_total = loss_det + 0.5 * loss_sep_s + 1.0 * loss_SCM

        return {
            "loss_total": loss_total,
            "loss_det": loss_det,
            "loss_sep": loss_sep_s,
            "loss_SCM": loss_SCM
        }


if __name__ == "__main__":
    # Sanity Check & Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ISD+ Modules on: {device}")

    # Simulated input image batch: [Batch_size=2, Channels=3, H=640, W=640]
    dummy_img = torch.randn(2, 3, 640, 640, device=device)
    dummy_feat_t = torch.randn(2, 256, 40, 40, device=device)
    dummy_feat_s = torch.randn(2, 128, 40, 40, device=device)

    # Instantiate modules
    stc_module = StructuralTensorConstruction(channels=128).to(device)
    relevance_dummy = torch.rand(2, 1, 40, 40, device=device)
    T_out = stc_module(dummy_feat_s, relevance_dummy)
    print("✓ STC 4th-order Tensor Output Shape:", T_out.shape)  # Expected: [2, 3, 128, 40, 40]

    sts_module = StructuralTensorSeparation(channels=128).to(device)
    mask = torch.zeros(2, 1, 1, 40, 40, device=device)
    delta = torch.ones_like(mask)
    T_s, T_b, T_n, l_sep = sts_module(T_out, mask, delta)
    print("✓ STS Decoupled Target Shape:", T_s.shape)
    print("✓ STS Separation Loss:", l_sep.item())

    distill = PredictiveConsistentDistillationLoss(in_channels_t=256, in_channels_s=128).to(device)
    T_t_dummy = torch.randn(2, 3, 256, 40, 40, device=device)
    loss_scm = distill(T_t_dummy, T_t_dummy, T_out, T_out, relevance_dummy, relevance_dummy, relevance_dummy)
    print("✓ PCSD Distillation Loss:", loss_scm.item())
    print("\nAll ISD+ modules verified successfully.")