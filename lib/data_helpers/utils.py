import os
import pickle

from tqdm import tqdm
from typing import List, Any

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
