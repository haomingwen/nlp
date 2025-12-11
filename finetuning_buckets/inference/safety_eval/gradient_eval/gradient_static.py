"""
Naive (single-process, single-GPU) version of gradient collection and projection.
No DeepSpeed, no distributed, just:
  - forward
  - backward
  - collect gradients into a vector
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails


def get_trak_projector(device: torch.device = torch.device("cuda:0")):
    """Select CUDA or basic projector depending on fast_jl availability."""
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector

def calc_loss(batch, outputs, tokenizer=None):
    labels = batch['labels']
    labels = labels[:, 1:].clone()
    if tokenizer is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    logits = outputs.logits[:, :-1, :]

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
    return loss

def obtain_gradients(model: nn.Module, tokenizer: AutoTokenizer, batch: dict) -> torch.Tensor:
    """Compute full parameter gradients for a single batch and return a flat CPU vector.

    Assumes:
      - single process, single GPU
      - batch tensors already on the same device as the model or CPU
    """
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    grad_chunks = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_chunks.append(p.grad.detach().view(-1).cpu())

    vectorized_grads = torch.cat(grad_chunks) if grad_chunks else torch.empty(0)
    # normalize the gradients
    normalized_grad = F.normalize(vectorized_grads, p=2, dim=-1)

    model.zero_grad()
    return vectorized_grads


def get_full_project_gradient(eval_dataloader: DataLoader, model: nn.Module, tokenizer: AutoTokenizer, project_interval: int = 10, proj_dim: int = 8192, dtype: torch.dtype = torch.float16, block_size: int = 1) -> torch.Tensor:
    """Compute gradients for several batches and stack them."""
    gradients_list = []
    projected_gradients = []
    device = "cuda:0"
    projector_cls = get_trak_projector(device=device)

    for i, batch in enumerate(
        tqdm(eval_dataloader, total=len(eval_dataloader), desc="Collecting gradients"),
        start=1,
    ):
        grad_vec = obtain_gradients(model, tokenizer, batch)  # (num_params,)
        gradients_list.append(grad_vec)
        # to prevent the memory overflow of all gradients, we project periodically
        if i % project_interval == 0:
            if gradients_list:
                current_gradient = torch.stack(gradients_list, dim=0)  # (project_interval, num_params)
            else:
                current_gradient = torch.empty(0, device=device)
            projected_gradient = project_gradient(current_gradient, projector_cls=projector_cls, proj_dim=proj_dim, device=device, dtype=dtype, block_size=block_size)
            projected_gradients.append(projected_gradient.cpu())

            gradients_list = []

    projected_gradients = torch.stack(projected_gradients, dim=0)   
    projected_gradients = projected_gradients.reshape(-1, projected_gradients.size(-1))
    # normalize gradients again
    normalized_gradients = F.normalize(projected_gradients, p=2, dim=-1)

    return normalized_gradients


def project_gradient(
    gradient: torch.Tensor,
    projector_cls: CudaProjector,
    proj_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    """Project high-dimensional gradient matrix to lower dimension with TRAK projector.

    gradient: (num_batches, num_params)
    returns: (num_batches, proj_dim) on CPU
    """
    proj = projector_cls(
        grad_dim=gradient.shape[1],
        proj_dim=proj_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=device,
        dtype=dtype,
        block_size=block_size,
        max_batch_size=8,
    )
    gradient = gradient.to(device=device, dtype=dtype)
    projected = proj.project(gradient, model_id=0)
    return projected.cpu()


def calculate_diversity(projected_gradients: torch.Tensor) -> torch.Tensor:
    """Compute a simple diversity metric over projected gradients.

    projected_gradients: (num_batches, proj_dim)
    """
    diversity = 0.0
    n = projected_gradients.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    for i in range(n):
        for j in range(i + 1, n):
            diversity += torch.abs(torch.dot(projected_gradients[i], projected_gradients[j]))

    diversity = diversity / (n * (n - 1) / 2)
    return diversity


def visualize_diversity(projected_gradients_a: torch.Tensor, projected_gradients_b: Optional[torch.Tensor] = None, out_path: str = "diversity.png") -> None:
    """Use PCA to reduce to 2D and save a scatter plot."""
    # 确保输出路径的目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # 如果目标路径是一个已存在的目录，则删除它
    if os.path.exists(out_path) and os.path.isdir(out_path):
        import shutil
        shutil.rmtree(out_path)
    
    if projected_gradients_b is None:
        projected_np = projected_gradients_a.numpy()
        pca = PCA(n_components=2)
        pca.fit(projected_np)
        projected_pca = pca.transform(projected_np)

        plt.figure()
        plt.scatter(projected_pca[:, 0], projected_pca[:, 1])
        plt.savefig(out_path)
        plt.close()
        return
    
    a_np = projected_gradients_a.numpy()
    b_np = projected_gradients_b.numpy()

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


class GradientStaticEvaluator:

    def __init__(self, model_path: str, out_path: str, proj_dim: int = 8192, block_size: int = 1):
        self.model_path = model_path
        self.out_path = out_path
        os.makedirs(out_path, exist_ok=True)
        self.proj_dim = proj_dim
        self.block_size = block_size
        self.dtype = torch.float16

    def init_engine(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            offload_state_dict=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self):
        from finetuning_buckets.inference.safety_eval.kl_eval.kl import (
            make_collate_fn,
            ConversationDataset,
        )
        from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails

        self.init_engine()
        self.model.train()

        # first project the unsafe data, then project the safe data
        safe_data, unsafe_data = get_beavertails(split="train")
        harmful_dataset = ConversationDataset(unsafe_data)

        eval_dataloader = DataLoader(
            harmful_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=make_collate_fn(self.tokenizer, use_chat=False, mask_prompts=True),
        )
        unsafe_tot = len(eval_dataloader)
        
        full_gradient_unsafe = get_full_project_gradient(eval_dataloader, self.model, self.tokenizer, project_interval=5, proj_dim=self.proj_dim, dtype=self.dtype, block_size=self.block_size)

        safe_dataset = ConversationDataset(safe_data)

        eval_dataloader = DataLoader(
            safe_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=make_collate_fn(self.tokenizer, use_chat=False, mask_prompts=True),
        )
        safe_tot = len(eval_dataloader)

        full_gradient_safe = get_full_project_gradient(eval_dataloader, self.model, self.tokenizer, project_interval=5, proj_dim=self.proj_dim, dtype=self.dtype, block_size=self.block_size)
        
        full_gradient = torch.cat((full_gradient_unsafe, full_gradient_safe), dim=0)

        safe_diversity = calculate_diversity(full_gradient_safe)
        unsafe_diversity = calculate_diversity(full_gradient_unsafe)
        print(f"safe_diversity: {safe_diversity}")
        print(f"unsafe_diversity: {unsafe_diversity}")

        output_path = os.path.join(self.out_path, f"diversity_samples_{unsafe_tot + safe_tot}.png")
        visualize_diversity(full_gradient_unsafe, full_gradient_safe, out_path=output_path)

        # store the results
        output_path = os.path.join(self.out_path, f"gradient_samples_unsafe_{unsafe_tot}.pt")
        torch.save(full_gradient_unsafe, output_path)

        output_path = os.path.join(self.out_path, f"gradient_samples_safe_{safe_tot}.pt")
        torch.save(full_gradient_safe, output_path)


if __name__ == "__main__":
    evaluator = GradientStaticEvaluator(model_path="/root/autodl-tmp/qwen2-1.5b", out_path="finetuning_buckets/inference/safety_eval/gradient_eval/qwen2/beavertails/train")
    evaluator.evaluate()