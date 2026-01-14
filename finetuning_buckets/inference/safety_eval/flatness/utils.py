import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def build_dataloader(tokenizer: AutoTokenizer, split: str, use_unsafe: bool, batch_size: int = 1) -> DataLoader:
    safe_data, unsafe_data = get_beavertails(split=split)
    data = unsafe_data if use_unsafe else safe_data
    dataset = ConversationDataset(data)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True, model_name="qwen"),
    )


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