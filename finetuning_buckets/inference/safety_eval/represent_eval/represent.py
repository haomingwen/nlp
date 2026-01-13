"""evaluate the representation diversity of datasets based on the last token of prompt"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from finetuning_buckets.inference.safety_eval.utils import ConversationDataset, calculate_diversity, visualize_diversity
from finetuning_buckets.datasets.utils.collate import make_collate_fn
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from finetuning_buckets.datasets.utils.get_eval_data import get_beavertails

# the function for getting representation for each data item
def get_representation(model, batch, chat_template: str = "qwen"):
    assert batch["input_ids"].shape[0] == 1, "only support batch size 1"
    with torch.no_grad():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1] # [B, L, H]
    # find the representation for the last token of the prompt
    last_hidden = last_hidden[0, :, :]
    labels = batch["labels"]
    mask = (labels == -100)
    # find the last token for the labels
    rep_tok = (~mask).nonzero(as_tuple=True)[1][0]
    # for qwen models, the last token is 6 tokens before the response begin.
    if chat_template == "qwen":
        prompt_tok = rep_tok - 6
    else:
        prompt_tok = rep_tok
    last_hidden = last_hidden[prompt_tok, :].clone()
        
    return last_hidden


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


def _gather_hiddens(
    hiddens: torch.Tensor,
    rank: int,
    world_size: int,
) -> Optional[torch.Tensor]:
    if not dist.is_initialized():
        return hiddens

    backend = dist.get_backend()
    if backend == "nccl":
        gather_device = torch.device("cuda", torch.cuda.current_device())
    else:
        gather_device = torch.device("cpu")

    hiddens = hiddens.to(gather_device)
    local_size = torch.tensor([hiddens.shape[0]], device=gather_device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(max(s.item() for s in size_list))

    if hiddens.shape[0] < max_size:
        pad = torch.zeros(
            (max_size - hiddens.shape[0], hiddens.shape[1]),
            device=gather_device,
            dtype=hiddens.dtype,
        )
        hiddens = torch.cat([hiddens, pad], dim=0)

    gather_list = [torch.zeros_like(hiddens) for _ in range(world_size)]
    dist.all_gather(gather_list, hiddens)

    if rank != 0:
        return None

    gathered = []
    for tensor, size in zip(gather_list, size_list):
        gathered.append(tensor[: int(size.item())].cpu())
    return torch.cat(gathered, dim=0)

# calculate represention diversity
# since the last embedding size is small, we don't need projection

# get the data and the model

if __name__ == "__main__":
    is_dist, rank, world_size, local_rank = _init_distributed()

    safe_data, unsafe_data = get_beavertails(split='train')
    safe_dataset = ConversationDataset(safe_data)
    unsafe_dataset = ConversationDataset(unsafe_data)

    model_path = "/root/autodl-tmp/qwen"
    device = torch.device("cuda", local_rank) if is_dist else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, offload_state_dict=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    safe_sampler = DistributedSampler(safe_dataset, shuffle=False) if is_dist else None
    safe_dataloader = DataLoader(
        safe_dataset,
        batch_size=1,
        shuffle=safe_sampler is None,
        sampler=safe_sampler,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True)
    )

    safe_hiddens = []
    for i, batch in enumerate(tqdm(safe_dataloader, total=len(safe_dataloader), desc="Collecting safe representation"),start=1,):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        hidden = get_representation(model, batch)
        safe_hiddens.append(hidden)

    if safe_hiddens:
        safe_hiddens = torch.stack(safe_hiddens, dim=0)
    else:
        safe_hiddens = torch.empty(0, model.config.hidden_size, dtype=torch.float32)
    safe_hiddens = _gather_hiddens(safe_hiddens, rank, world_size)

    unsafe_sampler = DistributedSampler(unsafe_dataset, shuffle=False) if is_dist else None
    unsafe_dataloader = DataLoader(
        unsafe_dataset,
        batch_size=1,
        shuffle=unsafe_sampler is None,
        sampler=unsafe_sampler,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True) 
    )
    unsafe_hiddens = []
    for i, batch in enumerate(tqdm(unsafe_dataloader, total=len(unsafe_dataloader), desc="Collecting unsafe representation"),start=1,):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        unsafe_hidden = get_representation(model, batch)
        unsafe_hiddens.append(unsafe_hidden)

    if unsafe_hiddens:
        unsafe_hiddens = torch.stack(unsafe_hiddens, dim=0)
    else:
        unsafe_hiddens = torch.empty(0, model.config.hidden_size, dtype=torch.float32)
    unsafe_hiddens = _gather_hiddens(unsafe_hiddens, rank, world_size)

    if is_dist and rank != 0:
        sys.exit(0)

    safe_diversity = calculate_diversity(safe_hiddens)
    unsafe_diversity = calculate_diversity(unsafe_hiddens)

    all_hiddens = [(safe_hiddens, "safe"), (unsafe_hiddens, "unsafe")]
    visualize_diversity(all_hiddens, out_path="finetuning_buckets/inference/safety_eval/represent_eval/qwen2/diversity.png")

    safe_tot = safe_hiddens.shape[0]
    unsafe_tot = unsafe_hiddens.shape[0]
    # store the hiddens for k-means 
    torch.save(safe_hiddens, f"finetuning_buckets/inference/safety_eval/represent_eval/qwen2/safe_hiddens_{safe_tot}.pt")
    torch.save(unsafe_hiddens, f"finetuning_buckets/inference/safety_eval/represent_eval/qwen2/unsafe_hiddens_{unsafe_tot}.pt")



