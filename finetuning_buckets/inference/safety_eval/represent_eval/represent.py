"""evaluate the representation diversity of datasets based on the last token of prompt"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from finetuning_buckets.inference.safety_eval.utils import ConversationDataset, calculate_diversity, visualize_diversity
from finetuning_buckets.inference.safety_eval.kl_eval.kl import make_collate_fn
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional

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

# calculate represention diversity
# since the last embedding size is small, we don't need projection

# get the data and the model

if __name__ == "__main__":
    safe_data, unsafe_data = get_beavertails(split='test')
    safe_dataset = ConversationDataset(safe_data)
    unsafe_dataset = ConversationDataset(unsafe_data)

    model_path = "/root/autodl-tmp/qwen2-1.5b"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16, offload_state_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    safe_dataloader = DataLoader(
        safe_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True)
    )

    safe_hiddens = []
    for i, batch in enumerate(tqdm(safe_dataloader, total=len(safe_dataloader), desc="Collecting safe representation"),start=1,):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        hidden = get_representation(model, batch)
        safe_hiddens.append(hidden)

    safe_hiddens = torch.stack(safe_hiddens, dim=0)
    safe_diversity = calculate_diversity(safe_hiddens)

    unsafe_dataloader = DataLoader(
        unsafe_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, mask_prompts=True) 
    )
    unsafe_hiddens = []
    for i, batch in enumerate(tqdm(unsafe_dataloader, total=len(unsafe_dataloader), desc="Collecting unsafe representation"),start=1,):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        unsafe_hidden = get_representation(model, batch)
        unsafe_hiddens.append(unsafe_hidden)

    unsafe_hiddens = torch.stack(unsafe_hiddens, dim=0)
    unsafe_diversity = calculate_diversity(unsafe_hiddens)

    all_hiddens = [(safe_hiddens, "safe"), (unsafe_hiddens, "unsafe")]
    visualize_diversity(all_hiddens, out_path="finetuning_buckets/inference/safety_eval/represent_eval/qwen2/diversity.png")

    safe_tot = safe_hiddens.shape[0]
    unsafe_tot = unsafe_hiddens.shape[0]
    # store the hiddens for k-means 
    torch.save(safe_hiddens, f"finetuning_buckets/inference/safety_eval/represent_eval/qwen2/safe_hiddens_{safe_tot}.pt")
    torch.save(unsafe_hiddens, f"finetuning_buckets/inference/safety_eval/represent_eval/qwen2/unsafe_hiddens_{unsafe_tot}.pt")





