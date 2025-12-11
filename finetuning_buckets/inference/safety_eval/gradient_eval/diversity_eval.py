# load the gradient tensor and the gradient clusters
import torch
import sys
import os
import json

grad_path = "gradient_samples_safe_1288.pt"
cluster_path = "finetuning_buckets/inference/safety_eval/gradient_eval/cluster_results/gradient_clusters_safe_1288_k32.pt"
grad_tensor = torch.load(grad_path, map_location="cpu")
cluster_info = torch.load(cluster_path, map_location="cpu")

cluster_labels = cluster_info["cluster_to_indices"]
k = cluster_info["k"]

total_sim = torch.zeros(k)
for i in range(k):
    indices = cluster_labels[i]
    # calculate the diversity of each gradient cluster
    gradients = [grad_tensor[idx] for idx in indices]
    for j in range(len(gradients)):
        for m in range(j+1, len(gradients)):
            sim = torch.abs(torch.dot(gradients[j], gradients[m]))
            total_sim[i] += sim
    total_sim[i] = total_sim[i] / (len(gradients) * (len(gradients) - 1) / 2)

out_dir = "finetuning_buckets/inference/safety_eval/gradient_eval/cluster_results"
cluster_json_path = os.path.join(out_dir, f"gradient_sim_safe_1288_k{k}.json")
with open(cluster_json_path, "w") as f:
    json.dump(
        {
            "k": k,
            "cluster_sims": total_sim.tolist(),
        },
        f,
        indent=2,
    )
print(f"Saved lightweight JSON summary to {cluster_json_path}")