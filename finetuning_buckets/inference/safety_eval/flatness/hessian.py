"""Compute top Hessian eigenvalues/eigenvectors for a HF causal LM using pyhessian."""

import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyhessian import hessian

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails
from finetuning_buckets.datasets.utils.collate import make_collate_fn
from finetuning_buckets.inference.safety_eval.utils import ConversationDataset


class InputIdsModelWrapper(nn.Module):
    """Wrap HF models so pyhessian can call forward with input_ids only."""

    def __init__(self, model: nn.Module, attention_mask: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask
        self.labels = labels

    def forward(self, input_ids: torch.Tensor):
        return self.model(
            input_ids=input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
        )


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


def build_dataloader(tokenizer: AutoTokenizer, split: str, use_unsafe: bool) -> DataLoader:
    safe_data, unsafe_data = get_beavertails(split=split)
    data = unsafe_data if use_unsafe else safe_data
    dataset = ConversationDataset(data)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True),
    )


def lm_loss(outputs, _targets):
    if not hasattr(outputs, "loss") or outputs.loss is None:
        raise ValueError("Model outputs must include loss; ensure labels are in the batch.")
    return outputs.loss


def compute_top_hessian_eigs(
    model_path: str,
    top_n: int = 1,
    split: str = "train",
    use_unsafe: bool = True,
    dtype: torch.dtype = torch.float16,
):
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype=dtype,
    )
    model.eval()

    dataloader = build_dataloader(tokenizer, split=split, use_unsafe=use_unsafe)
    batch = next(iter(dataloader))
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    inputs = batch["input_ids"]
    targets = batch["labels"]
    wrapped_model = InputIdsModelWrapper(
        model,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    hess = hessian(
        wrapped_model,
        lm_loss,
        data=(inputs, targets),
        cuda=torch.cuda.is_available(),
    )
    eigenvalues, eigenvectors = hess.eigenvalues(top_n=top_n)
    return eigenvalues, eigenvectors


def main():
    parser = argparse.ArgumentParser(description="Compute top Hessian eigenpairs with pyhessian.")
    parser.add_argument("--model-path", default="/root/autodl-tmp/qwen")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--split", default="train")
    parser.add_argument("--use-safe", action="store_true")
    args = parser.parse_args()

    use_unsafe = not args.use_safe
    eigenvalues, eigenvectors = compute_top_hessian_eigs(
        model_path=args.model_path,
        top_n=args.top_n,
        split=args.split,
        use_unsafe=use_unsafe,
    )

    print("Top eigenvalues:")
    for idx, val in enumerate(eigenvalues):
        print(f"{idx}: {val}")
    print("Top eigenvectors count:", len(eigenvectors))


if __name__ == "__main__":
    main()
