"""Visualize optimizer trajectory and loss surface via PCA directions."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails
from finetuning_buckets.datasets.utils.collate import make_collate_fn
from finetuning_buckets.inference.safety_eval.utils import ConversationDataset


def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
) -> Tuple[nn.Module, AutoTokenizer]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        offload_state_dict=True,
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_dataloader(
    tokenizer: AutoTokenizer, split: str, use_unsafe: bool, batch_size: int = 1
) -> DataLoader:
    safe_data, unsafe_data = get_beavertails(split=split)
    data = unsafe_data if use_unsafe else safe_data
    dataset = ConversationDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True),
    )


def _load_state_dict(ckpt_path: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    if os.path.isdir(ckpt_path):
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map=None,
        )
        return model.state_dict()
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def _collect_param_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k, v in state_dict.items() if torch.is_floating_point(v)]


def _vectorize_state_dict(state_dict: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    parts = [state_dict[k].detach().float().reshape(-1) for k in keys]
    return torch.cat(parts, dim=0)


def _devectorize_state_dict(
    vector: torch.Tensor,
    reference: Dict[str, torch.Tensor],
    keys: List[str],
) -> Dict[str, torch.Tensor]:
    new_state = {}
    idx = 0
    for key in keys:
        ref = reference[key]
        numel = ref.numel()
        chunk = vector[idx : idx + numel].view_as(ref).to(ref.dtype)
        new_state[key] = chunk
        idx += numel
    return new_state


@torch.no_grad()
def _compute_loss(
    model: nn.Module, dataloader: DataLoader, max_batches: int
) -> float:
    device = next(model.parameters()).device
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.detach().float().item())
    return float(sum(losses) / max(len(losses), 1))


def compute_pca_directions(vectors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.stack(vectors, dim=0)  # (n, p)
    _, _, v = torch.pca_lowrank(matrix, q=2, center=False)
    dir1 = v[:, 0]
    dir2 = v[:, 1]
    return dir1, dir2


def main():
    parser = argparse.ArgumentParser(
        description="Plot optimizer trajectory and loss surface along PCA directions."
    )
    parser.add_argument("--model-path", default="/root/autodl-tmp/qwen")
    parser.add_argument("--ckpt-list", required=True, help="Text file with checkpoint paths.")
    parser.add_argument("--out-dir", default="opt_results")
    parser.add_argument("--split", default="train")
    parser.add_argument("--use-safe", action="store_true")
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ckpt_list, "r", encoding="utf-8") as f:
        ckpt_paths = [line.strip() for line in f if line.strip()]
    if len(ckpt_paths) < 2:
        raise ValueError("Need at least two checkpoints for PCA.")

    reference_sd = _load_state_dict(ckpt_paths[-1], dtype=dtype)
    param_keys = _collect_param_keys(reference_sd)

    vectors = []
    for ckpt in ckpt_paths:
        state_dict = _load_state_dict(ckpt, dtype=dtype)
        vec = _vectorize_state_dict(state_dict, param_keys)
        vectors.append(vec)

    theta_n = vectors[-1]
    diffs = [v - theta_n for v in vectors]
    dir1, dir2 = compute_pca_directions(diffs)

    coords = torch.stack(
        [torch.stack((d.dot(dir1), d.dot(dir2))) for d in diffs], dim=0
    )

    model, tokenizer = load_model_and_tokenizer(args.model_path, dtype=dtype)
    dataloader = build_dataloader(
        tokenizer, split=args.split, use_unsafe=not args.use_safe, batch_size=args.batch_size
    )

    lin = torch.linspace(-args.scale, args.scale, args.grid)
    surface = torch.zeros(args.grid, args.grid)
    for i, x in enumerate(lin):
        for j, y in enumerate(lin):
            vec = theta_n + x * dir1 + y * dir2
            new_state = _devectorize_state_dict(vec, reference_sd, param_keys)
            model.load_state_dict(new_state, strict=False)
            surface[i, j] = _compute_loss(model, dataloader, args.max_batches)

    torch.save(
        {
            "coords": coords.cpu(),
            "surface": surface.cpu(),
            "lin": lin.cpu(),
        },
        os.path.join(args.out_dir, "opt_surface.pt"),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(lin.numpy(), lin.numpy(), surface.numpy(), levels=30, cmap="viridis")
    ax.plot(coords[:, 0].numpy(), coords[:, 1].numpy(), color="cyan", lw=2, marker="o")
    ax.set_xlabel("1st PCA component")
    ax.set_ylabel("2nd PCA component")
    fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "opt_surface.png"), dpi=200)


if __name__ == "__main__":
    main()
