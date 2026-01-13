from typing import Any, Callable, Dict, List, Optional, Union

import torch


def _extract_messages(sample: Union[Dict[str, Any], List[Dict[str, Any]], str]):
    if isinstance(sample, dict) and "messages" in sample:
        return sample["messages"]
    if isinstance(sample, list):
        return sample
    if isinstance(sample, str):
        return sample
    raise ValueError("Unsupported sample type for collate.")


def _format_messages_with_template(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool = False) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _build_text(tokenizer, messages: List[Dict[str, Any]], use_chat: bool) -> str:
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        return _format_messages_with_template(tokenizer, messages, add_generation_prompt=False)
    return _format_messages_fallback(messages)


def _build_prompt_text(tokenizer, messages: List[Dict[str, Any]], use_chat: bool) -> str:
    prompt_messages = [dict(m) for m in messages]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages[-1]["content"] = ""
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        return _format_messages_with_template(tokenizer, prompt_messages, add_generation_prompt=False)
    return _format_messages_fallback(prompt_messages)


def make_collate_fn(
    tokenizer,
    use_chat: bool = True,
    mask_prompts: bool = False,
    max_length: Optional[int] = None,
) -> Callable[[List[Any]], Dict[str, torch.Tensor]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for sample in batch:
            messages = _extract_messages(sample)
            if isinstance(messages, str):
                full_text = messages
                prompt_len = 0
            else:
                full_text = _build_text(tokenizer, messages, use_chat=use_chat)
                prompt_len = 0
                if mask_prompts:
                    prompt_text = _build_prompt_text(tokenizer, messages, use_chat=use_chat)
                    prompt_len = len(
                        tokenizer(prompt_text, add_special_tokens=False).input_ids
                    )

            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            if mask_prompts and prompt_len > 0:
                prompt_len = min(prompt_len, labels.shape[0])
                labels[:prompt_len] = -100

            if max_length is not None and input_ids.shape[0] > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        max_len = max(t.shape[0] for t in input_ids_list)
        if max_length is not None:
            max_len = min(max_len, max_length)

        def _pad(t: torch.Tensor, value: int) -> torch.Tensor:
            if t.shape[0] == max_len:
                return t
            pad_amount = max_len - t.shape[0]
            return torch.nn.functional.pad(t, (0, pad_amount), value=value)

        input_ids = torch.stack([_pad(t, pad_token_id) for t in input_ids_list])
        attention_mask = torch.stack([_pad(t, 0) for t in attention_mask_list])
        labels = torch.stack([_pad(t, -100) for t in labels_list])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn
