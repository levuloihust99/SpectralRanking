import json
import torch
from typing import Text, Dict, List, Any, Callable, Optional

from torch.utils.data import DataLoader

from lib.data_helpers.bytedataset import ByteDataset


def encode_texts(tokenizer, article: Text, summary: Text, sep_token, eos_token) -> List[int]:
    article_tokens = tokenizer.tokenize(article)
    summary_tokens = tokenizer.tokenize(summary)
    input_tokens = article_tokens + [sep_token] + summary_tokens + [eos_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    return input_ids


def do_pad(batch_input_ids: List[List[int]], pad_token_id: int) -> Dict[Text, List[List[int]]]:
    max_len = max([len(input_ids) for input_ids in batch_input_ids])
    padded_batch_input_ids = []
    batch_attn_mask = []
    for input_ids in batch_input_ids:
        pad_len = max_len - len(input_ids)
        attn_mask = [1] * len(input_ids) + [0] * pad_len
        batch_attn_mask.append(attn_mask)
        padded_input_ids = input_ids + [pad_token_id] * pad_len
        padded_batch_input_ids.append(padded_input_ids)

    return {
        "input_ids": padded_batch_input_ids,
        "attention_mask": batch_attn_mask
    }


def get_collate_fn(
    tokenizer,
    sep_token,
    max_input_len: Optional[int] = None
):
    def collate_fn(items: List[Dict[Text, Any]]):
        outputs = []
        for item in items:
            involved_summaries = {}
            for k, v in item["comparisons"].items():
                preferred = v.get("preferred")
                if preferred == 1 or preferred == -1:
                    g1, g2 = eval(k)
                    if g1 not in involved_summaries:
                        involved_summaries[g1] = item["summaries"][g1]
                    if g2 not in involved_summaries:
                        involved_summaries[g2] = item["summaries"][g2]

            if not involved_summaries:
                continue

            pretokenized = item.get("pretokenized")
            generators = []
            item_inputs_ids = []
            comparisons = {k: v.get("preferred") for k, v in item["comparisons"].items() if v.get("preferred") in {1, -1}}
            for g, summary in involved_summaries.items():
                generators.append(g)
                if pretokenized is None:
                    g_input_ids = encode_texts(tokenizer, item["article"], summary, sep_token=sep_token, eos_token=tokenizer.eos_token)
                else:
                    article_input_ids = pretokenized["article"]
                    if max_input_len:
                        article_input_ids = article_input_ids[:max_input_len]
                    g_input_ids = (
                        article_input_ids + tokenizer.convert_tokens_to_ids([sep_token]) +
                        pretokenized["summaries"][g] + [tokenizer.eos_token_id]
                    )
                item_inputs_ids.append(g_input_ids)

            item_inputs = do_pad(item_inputs_ids, pad_token_id=tokenizer.pad_token_id)
            outputs.append({"generators": generators, "inputs": item_inputs, "comparisons": comparisons})

        return outputs

    return collate_fn
