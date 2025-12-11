import os
import sys

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def calculate_diversity(vecs: torch.Tensor) -> torch.Tensor:
    """Compute a simple diversity metric over vectors.

    shape: (num_vecs, hidden_dims)
    """
    diversity = 0.0
    n = vecs.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    for i in range(n):
        for j in range(i + 1, n):
            diversity += torch.abs(torch.dot(vecs[i], vecs[j]))

    diversity = diversity / (n * (n - 1) / 2)
    return diversity

def visualize_diversity(
    vecs_set: List[Tuple[torch.Tensor, str]],
    out_path: str = "diversity.png",
    pca_dim: int = 2,
) -> None:
    """
    Visualize multiple sets of vectors in a shared PCA space.

    Args:
        vecs_set: list of (tensor, label_str).
                  Each tensor shape: (N_i, D), all D must be the same.
        out_path: path to save the figure.
        pca_dim:  PCA dimension (must be >= 2 if you want 2D scatter).
    """
    assert pca_dim >= 2, "pca_dim must be at least 2 for 2D visualization."

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path) and os.path.isdir(out_path):
        import shutil
        shutil.rmtree(out_path)

    # 每组大小
    set_sizes = [t.shape[0] for (t, _) in vecs_set]

    # 转成 numpy 并拼接
    all_vecs_np = [t.detach().cpu().numpy() for (t, _) in vecs_set]
    all_np = np.concatenate(all_vecs_np, axis=0)  # (sum N_i, D)

    # PCA
    pca = PCA(n_components=pca_dim)
    all_pca = pca.fit_transform(all_np)  # (sum N_i, pca_dim)

    # 画图
    plt.figure()
    tot = 0
    for (size, (_, label)) in zip(set_sizes, vecs_set):
        cur_pca = all_pca[tot:tot + size]
        plt.scatter(cur_pca[:, 0], cur_pca[:, 1], alpha=0.7, label=label)
        tot += size

    plt.legend()
    plt.title("Diversity in shared PCA space")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()





    
