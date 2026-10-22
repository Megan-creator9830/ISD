import torch
import torch.nn as nn
import torch.nn.functional as F

class ISDDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, tau=1.0, K=6, 
                 lambda_s_kd=1.0, lambda_b_kd=0.5, lambda_n_kd=0.5, lambda_R=0.5):
        super().__init__()
        self.alpha = alpha                # 论文公式 (6) 平衡系数
        self.tau = tau                    # 论文公式 (9) 温度系数
        self.K = K                        # 论文公式 (10) 尺度聚合范围 K=6
        self.lambda_s_kd = lambda_s_kd    # 论文公式 (22)
        self.lambda_b_kd = lambda_b_kd    # 论文公式 (22)
        self.lambda_n_kd = lambda_n_kd    # 论文公式 (25)
        self.lambda_R = lambda_R          # 论文公式 (25)
        self.eps = 1e-7

    def compute_prediction_relevance(self, feat, cls_pred, box_loss):

        p_q = torch.sigmoid(cls_pred)
        s_q = (p_q ** self.alpha) * torch.exp(-(1 - self.alpha) * box_loss)

        grad = torch.autograd.grad(s_q.sum(), feat, retain_graph=True, create_graph=True)[0]
        relevance = torch.abs(feat * grad)
        norm_factor = torch.sum(relevance, dim=[1, 2, 3], keepdim=True) + self.eps
        R = relevance / norm_factor
        return R, s_q

    def structure_tensor_construction(self, feat, R):

        B, C, H, W = feat.shape
        structure_slices = []
        for r in range(1, self.K + 1):
            kernel_size = 2 * r + 1
            unfold_feat = F.unfold(feat, kernel_size=kernel_size, padding=r)
            diff = unfold_feat - feat.view(B, C, -1).repeat(1, kernel_size**2, 1)
            dist = torch.norm(diff, p=2, dim=1, keepdim=True)
            k = torch.exp(-dist / self.tau)

            unfold_R = F.unfold(R, kernel_size=kernel_size, padding=r)
            numerator = (k * unfold_R * unfold_feat).sum(dim=1).view(B, C, H, W)
            denominator = (k * unfold_R).sum(dim=1).view(B, C, H, W) + self.eps
            O_r = numerator / denominator
            structure_slices.append(O_r.unsqueeze(-1))
            
        T = torch.cat(structure_slices, dim=-1) # B x C x H x W x K
        return T

    def structure_tensor_separation(self, T, R, gt_mask):
    
        A = torch.norm(T, p=2, dim=-1, keepdim=True)
        A = A / (torch.max(A) + self.eps)
        
        W_s = gt_mask.unsqueeze(-1) * R.unsqueeze(-1)
        W_b = (1.0 - gt_mask.unsqueeze(-1)) * R.unsqueeze(-1)
        W_n = A * (1.0 - R.unsqueeze(-1))

        T_s = T * W_s
        T_b = T * W_b
        T_n = T * W_n
        return T_s, T_b, T_n, W_n

    def pcsd_loss(self, T_s_T, T_s_S, T_b_T, T_b_S, W_n_T, R_S, R_T):
    
        loss_s = torch.norm(T_s_T - T_s_S, p='fro')**2 / (torch.norm(T_s_T, p='fro')**2 + self.eps)
        loss_b = torch.norm(T_b_T - T_b_S, p='fro')**2 / (torch.norm(T_b_T, p='fro')**2 + self.eps)
        L_v = self.lambda_s_kd * loss_s + self.lambda_b_kd * loss_b

        L_n = torch.sum(W_n_T.squeeze(-1) * R_S) / (torch.norm(W_n_T, p=1) + self.eps)

        cos_sim = torch.sum(R_T * R_S) / (torch.norm(R_T, p=2) * torch.norm(R_S, p=2) + self.eps)
        L_R = 1.0 - cos_sim

        L_PCSD = L_v + self.lambda_n_kd * L_n + self.lambda_R * L_R
        return L_PCSD
