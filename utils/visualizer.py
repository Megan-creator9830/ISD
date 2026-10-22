import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_structural_response_surface(tensor_slice, save_path="response_surface.png"):
  
    if isinstance(tensor_slice, torch.Tensor):
        data = tensor_slice.squeeze().detach().cpu().numpy()
    else:
        data = tensor_slice

    H, W = data.shape
    X = np.linspace(-4, 4, W)
    Y = np.linspace(-4, 4, H)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('$\Delta W$')
    ax.set_ylabel('$\Delta H$')
    ax.set_zlabel('Energy Density')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
