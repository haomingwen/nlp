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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trak.projectors import CudaProjector
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm
from typing import Optional, Callable, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails
from finetuning_buckets.datasets.utils.collate import make_collate_fn
from finetuning_buckets.inference.safety_eval.utils import (
    ConversationDataset,
    calculate_diversity,
    visualize_diversity,
)
from finetuning_buckets.inference.safety_eval.gradient_eval.gradient_utils import (
    get_trak_projector,
    get_fastjl_projector,
    project,
)

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


def _init_distributed() -> Tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 1, 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def _gather_projected(
    projected: torch.Tensor,
    rank: int,
    world_size: int,
) -> Optional[torch.Tensor]:
    if not dist.is_initialized():
        return projected

    backend = dist.get_backend()
    if backend == "nccl":
        gather_device = torch.device("cuda", torch.cuda.current_device())
    else:
        gather_device = torch.device("cpu")

    projected = projected.to(gather_device)
    local_size = torch.tensor([projected.shape[0]], device=gather_device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(max(s.item() for s in size_list))

    if projected.shape[0] < max_size:
        pad = torch.zeros(
            (max_size - projected.shape[0], projected.shape[1]),
            device=gather_device,
            dtype=projected.dtype,
        )
        projected = torch.cat([projected, pad], dim=0)

    gather_list = [torch.zeros_like(projected) for _ in range(world_size)]
    dist.all_gather(gather_list, projected)

    if rank != 0:
        return None

    gathered = []
    for tensor, size in zip(gather_list, size_list):
        gathered.append(tensor[: int(size.item())].cpu())
    return torch.cat(gathered, dim=0)


def get_full_project_gradient(
    eval_dataloader: DataLoader,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    project_interval: int = 10,
    proj_dim: int = 8192,
    dtype: torch.dtype = torch.float16,
    block_size: int = 1,
    projector_type: str = "trak",
    gather_to_rank0: bool = True,
) -> Optional[torch.Tensor]:
    """Compute gradients for several batches and stack them."""
    gradients_list = []
    projected_gradients = []
    device = next(model.parameters()).device
    projector_cls = None
    fastjl_projector = None
    if projector_type == "trak":
        projector_cls = get_trak_projector(device=device)
    elif projector_type == "fastjl":
        fastjl_projector = get_fastjl_projector(device=device)
    else:
        raise ValueError(f"Unknown projector_type: {projector_type}")

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
            projected_gradient = project(
                current_gradient,
                projector_cls=projector_cls,
                proj_dim=proj_dim,
                device=device,
                dtype=dtype,
                block_size=block_size,
                projector_type=projector_type,
                fastjl_projector=fastjl_projector,
            )
            projected_gradients.append(projected_gradient.cpu())

            gradients_list = []

    if not projected_gradients:
        projected_gradients = torch.empty(0, proj_dim)
    else:
        projected_gradients = torch.stack(projected_gradients, dim=0)
        projected_gradients = projected_gradients.reshape(-1, projected_gradients.size(-1))
    # normalize gradients again
    normalized_gradients = F.normalize(projected_gradients, p=2, dim=-1)

    if dist.is_initialized() and gather_to_rank0:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return _gather_projected(normalized_gradients, rank, world_size)

    return normalized_gradients



class GradientStaticEvaluator:

    def __init__(self, model_path: str, out_path: str, proj_dim: int = 8192, block_size: int = 1):
        self.model_path = model_path
        self.out_path = out_path
        os.makedirs(out_path, exist_ok=True)
        self.proj_dim = proj_dim
        self.block_size = block_size
        self.dtype = torch.float16

    def init_engine(self):
        is_dist, rank, world_size, local_rank = _init_distributed()
        if is_dist:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            offload_state_dict=True,
        )
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self):
        is_dist, rank, world_size, local_rank = _init_distributed()
        self.init_engine()
        self.model.train()

        # first project the unsafe data, then project the safe data
        safe_data, unsafe_data = get_beavertails(split="train")
        harmful_dataset = ConversationDataset(unsafe_data)
        harmful_sampler = None
        if is_dist:
            harmful_sampler = DistributedSampler(harmful_dataset, shuffle=False)

        eval_dataloader = DataLoader(
            harmful_dataset,
            batch_size=1,
            shuffle=harmful_sampler is None,
            sampler=harmful_sampler,
            collate_fn=make_collate_fn(self.tokenizer, mask_prompts=True),
        )
        unsafe_tot = len(harmful_dataset)
        
        full_gradient_unsafe = get_full_project_gradient(
            eval_dataloader,
            self.model,
            self.tokenizer,
            project_interval=20,
            proj_dim=self.proj_dim,
            dtype=self.dtype,
            block_size=self.block_size,
        )

        safe_dataset = ConversationDataset(safe_data)
        safe_sampler = None
        if is_dist:
            safe_sampler = DistributedSampler(safe_dataset, shuffle=False)

        eval_dataloader = DataLoader(
            safe_dataset,
            batch_size=1,
            shuffle=safe_sampler is None,
            sampler=safe_sampler,
            collate_fn=make_collate_fn(self.tokenizer, mask_prompts=True),
        )
        safe_tot = len(safe_dataset)

        full_gradient_safe = get_full_project_gradient(
            eval_dataloader,
            self.model,
            self.tokenizer,
            project_interval=20,
            proj_dim=self.proj_dim,
            dtype=self.dtype,
            block_size=self.block_size,
        )

        if is_dist and rank != 0:
            return
        
        full_gradient = torch.cat((full_gradient_unsafe, full_gradient_safe), dim=0)

        safe_diversity = calculate_diversity(full_gradient_safe)
        unsafe_diversity = calculate_diversity(full_gradient_unsafe)
        print(f"safe_diversity: {safe_diversity}")
        print(f"unsafe_diversity: {unsafe_diversity}")

        output_path = os.path.join(self.out_path, f"diversity_samples_{unsafe_tot + safe_tot}.png")
        visualize_diversity(
            projected_gradients_a=full_gradient_unsafe,
            projected_gradients_b=full_gradient_safe,
            out_path=output_path,
        )

        # store the results
        output_path = os.path.join(self.out_path, f"gradient_samples_unsafe_{unsafe_tot}.pt")
        torch.save(full_gradient_unsafe, output_path)

        output_path = os.path.join(self.out_path, f"gradient_samples_safe_{safe_tot}.pt")
        torch.save(full_gradient_safe, output_path)


if __name__ == "__main__":
    evaluator = GradientStaticEvaluator(model_path="/root/autodl-tmp/qwen", out_path="finetuning_buckets/inference/safety_eval/gradient_eval/qwen2/beavertails")
    evaluator.evaluate()
