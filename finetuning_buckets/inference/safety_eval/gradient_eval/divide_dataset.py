import os
import sys
import json
from typing import Optional, List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, Subset


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails
from finetuning_buckets.inference.safety_eval.utils import serialize

def divide_dataset_indices(indices: List[int], target: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    split_path = os.path.join(out_dir, f"split.json")

    split = Subset(target, indices)
    data = []
    for j in range(len(split)):
        item = split[j]
        data.append(serialize(item))
    print(f"dumping dataset subset with size {len(indices)}")
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(
            [
                {
                    'data': data,
                    'n': len(split),
                }
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved data split to {out_dir}")
def divide_dataset_clusters(km_vec: torch.Tensor, target: Dataset, out_dir: str) -> None:
    """
    split the dataset based on clustering result
    """
    os.makedirs(out_dir, exist_ok=True)

    k = km_vec["k"]
    sims = km_vec.get("sims", "")
    if sims is not None:
        sims = serialize(sims)
    k_means_center = km_vec.get("kmeans_centers", "")
    if k_means_center is not None:
        k_means_center = serialize(k_means_center)
        
    for i in range(k):
        split_path = os.path.join(out_dir, f"split_{i}.json")
        indices = km_vec["cluster_to_indices"][i]
        split = Subset(target, indices)
        # save as json
        data = []
        for j in range(len(split)):
            item = split[j]
            data.append(serialize(item))
        print(f"dumping the {i}-th cluster with sim {sims[i]} and size {len(split)}")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(
                [
                    {
                        'data': data,
                        'sim': sims[i],
                        'k_means_center': k_means_center[i],
                        'n': len(split),
                        'original_indices': indices,
                    }
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved data splits to {out_dir}")

if __name__ == "__main__":
    kmeans_path = "/root/autodl-tmp/shallow-vs-deep-alignment-backup/finetuning_buckets/inference/safety_eval/gradient_eval/qwen2/beavertails/train/k_means/clusters_samples11600.pt"
    grad_tensor = torch.load(kmeans_path, map_location="cpu")
    safe_dataset, _ = get_beavertails(split="train")
    divide_dataset_clusters(km_vec=grad_tensor, target=safe_dataset, out_dir="finetuning_buckets/inference/safety_eval/gradient_eval/dataset_splits_safe_new")











