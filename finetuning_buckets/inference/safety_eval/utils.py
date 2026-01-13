import os
import sys

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
def serialize(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()  
    elif isinstance(x, (int, float, str, bool)) or x is None:
        return x
    elif isinstance(x, dict):
        return {k: serialize(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [serialize(v) for v in x]
    else:
        return str(x) 

def calculate_diversity(vecs: torch.Tensor, use_abs: bool = True) -> torch.Tensor:
    """Compute a simple diversity metric over vectors.

    shape: (num_vecs, hidden_dims)
    """
    diversity = 0.0
    n = vecs.shape[0]
    if n < 2:
        return torch.tensor(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            if use_abs:
                diversity += torch.abs(torch.dot(vecs[i], vecs[j]))
            else:
                diversity += torch.dot(vecs[i], vecs[j])

    diversity = diversity / (n * (n - 1) / 2)
    return diversity

def visualize_diversity(
    vecs_set: Optional[List[Tuple[torch.Tensor, str]]] = None,
    set_num: Optional[int] = None,
    out_path: str = "diversity.png",
    pca_dim: int = 2,
    projected_gradients_a: Optional[torch.Tensor] = None,
    projected_gradients_b: Optional[torch.Tensor] = None,
) -> None:
    """
    Visualize vectors in a shared PCA space.

    Modes:
      1) Labeled vectors: pass vecs_set (optionally set_num).
      2) Gradient sets: pass projected_gradients_a (and optionally projected_gradients_b).
    """
    assert pca_dim >= 2, "pca_dim must be at least 2 for 2D visualization."

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path) and os.path.isdir(out_path):
        import shutil
        shutil.rmtree(out_path)

    if projected_gradients_a is not None:
        if projected_gradients_b is None:
            projected_np = projected_gradients_a.detach().cpu().numpy()
            pca = PCA(n_components=2)
            projected_pca = pca.fit_transform(projected_np)

            plt.figure()
            plt.scatter(projected_pca[:, 0], projected_pca[:, 1])
            plt.savefig(out_path)
            plt.close()
            return

        a_np = projected_gradients_a.detach().cpu().numpy()
        b_np = projected_gradients_b.detach().cpu().numpy()
        all_np = np.concatenate([a_np, b_np], axis=0)

        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_np)

        n_a = a_np.shape[0]
        a_pca = all_pca[:n_a]
        b_pca = all_pca[n_a:]

        plt.figure()
        plt.scatter(a_pca[:, 0], a_pca[:, 1], alpha=0.7, label="A")
        plt.scatter(b_pca[:, 0], b_pca[:, 1], alpha=0.7, label="B")
        plt.legend()
        plt.savefig(out_path)
        plt.close()
        return

    if vecs_set is None:
        raise ValueError("vecs_set is required when projected_gradients_a is not provided.")

    label_to_idx = {}
    if set_num is None:
        for _, label in vecs_set:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
        set_num = len(label_to_idx)

    vec_bins = [torch.empty(0, vecs_set[0][0].shape[-1]) for _ in range(set_num)]
    for vecs in vecs_set:
        vec, label = vecs
        if label in label_to_idx:
            label_idx = label_to_idx[label]
        elif isinstance(label, (int, np.integer)) and 0 <= int(label) < set_num:
            label_idx = int(label)
        else:
            if len(label_to_idx) >= set_num:
                raise ValueError("set_num is smaller than the number of labels.")
            label_to_idx[label] = len(label_to_idx)
            label_idx = label_to_idx[label]

        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        elif vec.dim() != 2:
            raise ValueError("vecs_set tensors must be 1D or 2D.")
        vec_bins[label_idx] = torch.cat((vec_bins[label_idx], vec), dim=0)

    set_sizes = [len(vec_bins[i]) for i in range(set_num)]
    vec_bins = torch.cat(vec_bins, dim=0)
    all_np = vec_bins.detach().cpu().numpy()

    pca = PCA(n_components=pca_dim)
    all_pca = pca.fit_transform(all_np)  # (sum N_i, pca_dim)

    plt.figure()
    tot = 0
    for i, size in enumerate(set_sizes):
        cur_pca = all_pca[tot:tot + size]
        plt.scatter(cur_pca[:, 0], cur_pca[:, 1], alpha=0.7, label=i)
        tot += size

    plt.legend()
    plt.title("Diversity in shared PCA space")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def k_means(vecs: torch.Tensor, k: int, out_path: Optional[str] = None, save_results: bool = False, inits: int = 10, iters: int = 100) -> List[List[int]]:
    """
        do k-means clustering to a set of vectors.
        shape: [N, num_hiddens]
        return the clustered indices of the data.
        save_path should be the directory.
    """
    n = vecs.shape[0]
    vecs_np = vecs.detach().cpu().numpy()
    kmeans = KMeans(
        n_clusters=k,
        n_init=inits,
        max_iter=iters
    )
    labels = kmeans.fit_predict(vecs_np)

    cluster_to_indices = {int(c): [] for c in range(k)}
    for vec_idx, label in zip([i for i in range(n)], labels.tolist()):
        cluster_to_indices[int(label)].append(int(vec_idx))

    # calculate diversity of each split
    sims = []
    for i in range(k):
        cluster_vecs = [vecs[cluster_to_indices[i][j]] for j in range(len(cluster_to_indices[i]))]
        cluster_vecs = torch.stack(cluster_vecs, dim=0)
        sims.append(calculate_diversity(cluster_vecs))
    
    if out_path is not None:
        # visualize the data
        figure_path = os.path.join(out_path, f"k_means_samples{n}.png")
        vec_comb = [(vecs[i], labels[i]) for i in range(n)]
        visualize_diversity(vecs_set=vec_comb, set_num=k, out_path=figure_path)
        print(f"Saved kmeans figure to {figure_path}")
        if save_results:
            cluster_pt_path = os.path.join(out_path, f"clusters_samples{n}.pt")
            torch.save(
                {
                    "k": k,
                    "cluster_labels": torch.as_tensor(labels),
                    "cluster_to_indices": cluster_to_indices,
                    "sims": torch.as_tensor(sims),
                    "kmeans_centers": torch.from_numpy(kmeans.cluster_centers_),
                },
                cluster_pt_path,
            )
            print(f"Saved clustered result to {cluster_pt_path}")
    
    return cluster_to_indices


def _compute_weighted_center(
    centers: torch.Tensor,
    cluster_sizes: dict,
    cluster_ids: List[int],
) -> torch.Tensor:
    sizes = torch.tensor([cluster_sizes[i] for i in cluster_ids], dtype=centers.dtype, device=centers.device)
    total = sizes.sum()
    if total == 0:
        return centers[cluster_ids].mean(dim=0)
    return (centers[cluster_ids] * sizes[:, None]).sum(dim=0) / total


def merge_kmeans_splits(
    cluster_to_indices: dict,
    kmeans_centers: torch.Tensor,
    merge_sizes: Optional[List[int]] = None,
    start_cluster: Optional[int] = None,
    max_merged_size: Optional[int] = 10000,
) -> Tuple[List[int], dict]:
    """
    Merge clusters into cumulative splits (e.g., sizes 1/2/4/8) by nearest-center order.

    Returns:
      ordered_clusters: the chosen cluster ordering
      merged_splits: dict[size] -> merged indices list for the first `size` clusters
    """
    if merge_sizes is None:
        merge_sizes = [1, 2, 4, 8]
    merge_sizes = sorted(set(merge_sizes))

    if not torch.is_tensor(kmeans_centers):
        kmeans_centers = torch.as_tensor(kmeans_centers)

    k = kmeans_centers.shape[0]
    max_size = merge_sizes[-1]
    if max_size > k:
        raise ValueError(f"merge size {max_size} exceeds number of clusters {k}")

    cluster_sizes = {int(i): len(cluster_to_indices[i]) for i in range(k)}

    if start_cluster is None:
        overall_center = kmeans_centers.mean(dim=0)
        dists = torch.norm(kmeans_centers - overall_center, dim=1)
        start_cluster = int(torch.argmin(dists).item())

    ordered = [start_cluster]
    remaining = set(range(k)) - {start_cluster}
    current_center = _compute_weighted_center(kmeans_centers, cluster_sizes, ordered)

    while len(ordered) < max_size:
        remaining_list = sorted(remaining)
        candidate_centers = kmeans_centers[remaining_list]
        dists = torch.norm(candidate_centers - current_center, dim=1)
        next_idx = remaining_list[int(torch.argmin(dists).item())]
        ordered.append(next_idx)
        remaining.remove(next_idx)
        current_center = _compute_weighted_center(kmeans_centers, cluster_sizes, ordered)

    merged_splits = {}
    for size in merge_sizes:
        per_cluster_limit = None
        if max_merged_size is not None:
            per_cluster_limit = max_merged_size // size
        merged = []
        for cluster_id in ordered[:size]:
            cluster_indices = cluster_to_indices[cluster_id]
            if per_cluster_limit is not None:
                merged.extend(cluster_indices[:per_cluster_limit])
            else:
                merged.extend(cluster_indices)
        merged_splits[size] = merged

    return ordered, merged_splits


    
