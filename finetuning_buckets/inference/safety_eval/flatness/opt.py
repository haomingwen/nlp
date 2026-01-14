"""Visualize optimizer trajectory and loss surface via PCA directions."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finetuning_buckets.inference.safety_eval.gradient_eval.gradient_utils import project, get_trak_projector
from finetuning_buckets.inference.safety_eval.flatness.utils import load_model_and_tokenizer, build_dataloader, _compute_loss

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ckpt_list, "r", encoding="utf-8") as f:
        ckpt_paths = [line.strip() for line in f if line.strip()]
    if len(ckpt_paths) < 2:
        raise ValueError("Need at least two checkpoints for PCA.")

    reference_sd = _load_state_dict(ckpt_paths[-1], dtype=dtype)
    param_keys = _collect_param_keys(reference_sd)

    theta_n = _vectorize_state_dict(reference_sd, param_keys)
    # project
    projector_cls = get_trak_projector(device=device)
    p_theta_n = project(
        theta_n.unsqueeze(0),
        projector_cls=projector_cls,
        proj_dim=8192,
        device=device,
        dtype=dtype,
        block_size=1,
    )

    diffs = []
    state_vecs = []
    for ckpt in ckpt_paths:
        state_dict = _load_state_dict(ckpt, dtype=dtype)
        vec = _vectorize_state_dict(state_dict, param_keys)
        # project
        p_vec = project(
            vec.unsqueeze(0),
            projector_cls=projector_cls,
            proj_dim=8192,
            device=device,
            dtype=dtype,
            block_size=1,
        )
        state_vecs.append(p_vec.squeeze(0))
        diffs.append((p_vec - p_theta_n).numpy())
        del state_dict, vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    diffs_np = np.vstack(diffs)

    pca = PCA(n_components=2)
    pca.fit(diffs_np)
    comps = torch.from_numpy(pca.components_.T).float()
    dir1, dir2 = comps[:, 0], comps[:, 1]

    # compute the coordinates of each ckpt
    coords = []
    ref_vec = state_vecs[-1]
    for vec in state_vecs:
        coor1, coor2 = (vec - ref_vec).dot(dir1), (vec - ref_vec).dot(dir2)
        coords.append([coor1.item(), coor2.item()])

    # compute the loss for each ckpt
    losses = []
    for ckpt in ckpt_paths:
        model, tokenizer = load_model_and_tokenizer(ckpt, dtype=dtype)
        dataloader = build_dataloader(
            tokenizer, split=args.split, use_unsafe=not args.use_safe, batch_size=args.batch_size
        )
        loss = _compute_loss(model, dataloader, args.max_batches)
        losses.append(loss)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # plot the trajectory
    fig, ax = plt.subplots(figsize=(7, 5))
    coords_np = np.asarray(coords, dtype=np.float32)
    losses_np = np.asarray(losses, dtype=np.float32)
    sc = ax.scatter(coords_np[:, 0], coords_np[:, 1], c=losses_np, cmap="viridis", s=80)
    ax.plot(coords_np[:, 0], coords_np[:, 1], color="black", lw=1.2, alpha=0.7)
    if coords_np.shape[0] > 0:
        ax.text(coords_np[0, 0], coords_np[0, 1], "start", fontsize=9, ha="left", va="bottom")
        ax.text(coords_np[-1, 0], coords_np[-1, 1], "end", fontsize=9, ha="left", va="bottom")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "coords_scatter.png"), dpi=200)

    # currently disabled, will be enabled when using hessian's eigenvector as the interpolation vector
    # lin = torch.linspace(-args.scale, args.scale, args.grid)
    # surface = torch.zeros(args.grid, args.grid)
    # for i, x in enumerate(lin):
    #     for j, y in enumerate(lin):
    #         vec = theta_n + x * dir1 + y * dir2
    #         new_state = _devectorize_state_dict(vec, reference_sd, param_keys)
    #         model.load_state_dict(new_state, strict=False)
    #         surface[i, j] = _compute_loss(model, dataloader, args.max_batches)

    # torch.save(
    #     {
    #         "surface": surface.cpu(),
    #         "lin": lin.cpu(),
    #     },
    #     os.path.join(args.out_dir, "opt_surface.pt"),
    # )

    # fig, ax = plt.subplots(figsize=(7, 5))
    # cs = ax.contourf(lin.numpy(), lin.numpy(), surface.numpy(), levels=30, cmap="viridis")
    # ax.plot(coords[:, 0].numpy(), coords[:, 1].numpy(), color="cyan", lw=2, marker="o")
    # ax.set_xlabel("1st PCA component")
    # ax.set_ylabel("2nd PCA component")
    # fig.colorbar(cs, ax=ax, shrink=0.8)
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.out_dir, "opt_surface.png"), dpi=200)


if __name__ == "__main__":
    main()
