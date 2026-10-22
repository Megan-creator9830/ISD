import torch
import numpy as np

def compute_strs(T1, T2):

    flat1 = T1.flatten().detach().cpu().numpy()
    flat2 = T2.flatten().detach().cpu().numpy()
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(flat1, flat2) / (norm1 * norm2))

def compute_eiou(pred_saliency, gt_box, img_shape=(640, 640)):

    H, W = img_shape
    mask = torch.zeros((H, W), dtype=torch.float32)
    x1, y1, x2, y2 = map(int, gt_box)
    mask[y1:y2, x1:x2] = 1.0
    
    saliency_energy = pred_saliency.squeeze()
    inside_energy = torch.sum(saliency_energy * mask)
    total_energy = torch.sum(saliency_energy) + 1e-7
    return float(inside_energy / total_energy)
