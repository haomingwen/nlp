import os
import sys
import json

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails
from finetuning_buckets.inference.safety_eval.utils import serialize

def divide_dataset(km_vec: torch.Tensor, target: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    k = km_vec["k"]
    sims = km_vec.get("sims", "")
    for i in range(k):
        split_path = os.path.join(out_dir, f"split_{i}.json")
        indices = km_vec["cluster_to_indices"][i]
        split = Subset(target, indices=indices)
        # save as json
        data = []
        for j in range(len(split)):
            item = split[j]
            data.append(serialize(item))
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(
                [
                    {
                        'data': data,
                        # 'sim': sims[i],
                        'n': len(split),
                    }
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved data splits to {out_dir}")

if __name__ == "__main__":
    kmeans_path = "finetuning_buckets/inference/safety_eval/gradient_eval/cluster_results_test/gradient_clusters_safe_1288_k8.pt"
    grad_tensor = torch.load(kmeans_path, map_location="cpu")
    safe_dataset, _ = get_beavertails(split="test")
    divide_dataset(km_vec=grad_tensor, target=safe_dataset, out_dir="finetuning_buckets/inference/safety_eval/gradient_eval/dataset_splits_test")











