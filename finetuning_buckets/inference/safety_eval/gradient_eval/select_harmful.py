"""select harmful responses based on the similarity with the k-means center"""
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
from finetuning_buckets.inference.safety_eval.gradient_eval.divide_dataset import divide_dataset_indices

class Metrics:
    SIMILAR = "similar"
    ORTHOGONAL = "orthogonal"
    OPPOSITE = "opposite"

def select_with_metrics(target: torch.Tensor, vecs: torch.Tensor, metric: str, keep_num: int = 1000):
    """select tensors based on the similarity.
    return both the target tensors and the indices selected.
    """

    sim_lst = vecs @ target
    idx_lst = torch.argsort(sim_lst, descending=True)

    if metric == Metrics.SIMILAR:
        return sim_lst[idx_lst[:keep_num]].mean(dim=-1), torch.stack([vecs[idx_lst[k]] for k in range(keep_num)], dim=0), idx_lst[:keep_num]
    
    if metric == Metrics.OPPOSITE:
        return sim_lst[idx_lst[-keep_num:]].mean(dim=-1), torch.stack([vecs[idx_lst[-k]] for k in range(1, keep_num + 1)], dim=0), idx_lst[-keep_num:]
    
    if metric == Metrics.ORTHOGONAL:
        sim_lst = torch.abs(sim_lst)
        idx_lst = torch.argsort(sim_lst, descending=True)
        return sim_lst[idx_lst[-keep_num:]].mean(dim=-1), torch.stack([vecs[idx_lst[-k]] for k in range(1, keep_num + 1)], dim=0), idx_lst[-keep_num:]

    else:
        raise NotImplementedError(f"metric {metric} is not implemented")

if __name__ == "__main__":
    k = 8
    metric = "similar"
    grad_path = "finetuning_buckets/inference/safety_eval/gradient_eval/qwen2/beavertails/train/gradient_samples_unsafe_15582.pt"
    splits_dir = "finetuning_buckets/inference/safety_eval/gradient_eval/dataset_splits_safe_new"
    out_dir = "finetuning_buckets/inference/safety_eval/gradient_eval/dataset_splits_unsafe"
    for i in range(k):
        # import the safe dataset split based on json
        split_path = os.path.join(splits_dir, f"split_{i}.json")
        with open(split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        target = torch.tensor(data[0]["k_means_center"])

        # import the unsafe vecs based on gradient file
        grad_tensor = torch.load(grad_path, map_location="cpu")

        sim, _, indices = select_with_metrics(target=target, vecs=grad_tensor, metric=metric)
        sim = sim.item()

        _, unsafe_dataset = get_beavertails(split="train")

        print(f"dumping unsafe split {i} with similarity {sim}")
        
        # select the dataset
        out_path = os.path.join(out_dir, f"split{i}")
        divide_dataset_indices(indices, unsafe_dataset, out_dir=out_path)
