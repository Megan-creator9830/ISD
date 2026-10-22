import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ISDStudentBackbone(nn.Module):
    """
    轻量化骨干网络，支持 ISD+L 与 ISD+XXS 配置 (Table 1)
    """
    def __init__(self, depth_mult=0.68, width_mult=0.60, base_dim=32):
        super().__init__()
        c1 = int(base_dim * width_mult)
        c2 = int(base_dim * 2 * width_mult)
        c3 = int(base_dim * 4 * width_mult)

        self.stem = ConvBNAct(3, c1, 3, 2, 1)
        self.stage1 = ConvBNAct(c1, c2, 3, 2, 1)
        self.stage2 = ConvBNAct(c2, c3, 3, 2, 1)
        self.out_dim = c3

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.stage1(x1)
        feat = self.stage2(x2)
        return feat

class ISDDetector(nn.Module):
    """
    轻量化检测器架构 (基于 RT-DETR / 解耦检测头)
    """
    def __init__(self, config='XXS', num_classes=1):
        super().__init__()
        if config == 'XXS':  # Table 1: d=0.68, w=0.60, dim=32, 3.3M Params
            self.backbone = ISDStudentBackbone(depth_mult=0.68, width_mult=0.60, base_dim=32)
        elif config == 'L':   # Table 1: d=1.02, w=1.08, dim=64, 8.6M Params
            self.backbone = ISDStudentBackbone(depth_mult=1.02, width_mult=1.08, base_dim=64)
        else:
            raise ValueError(f"Unsupported config: {config}")

        dim = self.backbone.out_dim
        # 解耦检测头：分类与边界框回归
        self.cls_head = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(dim, 4, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        cls_logits = self.cls_head(feat)
        bbox_preds = self.reg_head(feat)
        return {
            'features': feat,
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds
        }

def build_isd_model(config='XXS'):
    return ISDDetector(config=config)
