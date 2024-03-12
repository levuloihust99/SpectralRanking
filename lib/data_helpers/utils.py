import os
import pickle
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
from typing import List, Any
from .bytedataset import ByteDataset

def create_bytedataset(output_dir, data: List[Any]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    idx_writer = open(os.path.join(output_dir, "idxs.pkl"), "wb")
    dataset_size_place_holder = (0).to_bytes(4, 'big', signed=False)
    idx_writer.write(dataset_size_place_holder)

    data_writer = open(os.path.join(output_dir, "data.pkl"), "wb")
    num_records = 0
    for item in tqdm(data):
        idx_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(item, data_writer)
        num_records += 1

    data_writer.close()
    idx_writer.seek(0, 0)
    idx_writer.write(num_records.to_bytes(4, 'big', signed=False))


def pretokenize(bytedataset_path, tokenizer):
    pretokenize_path = os.path.join(bytedataset_path, "pretokenized")
    if os.path.exists(pretokenize_path):
        return

    dataset = ByteDataset(bytedataset_path)
    data = []
    for item in tqdm(dataset, desc="Pretokenizing"):
        article = item["article"]
        article_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article))
        summaries_tokens_ids = {}
        for k, v in item["summaries"].items():
            s_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v))
            summaries_tokens_ids[k] = s_tokens_ids
        pretokenized = {
            "article": article_tokens_ids,
            "summaries": summaries_tokens_ids
        }
        data.append({**item, "pretokenized": pretokenized})

    create_bytedataset(pretokenize_path, data)
