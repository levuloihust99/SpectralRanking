"""Define ByteDataset class.

ByteDataset is used to load big dataset and easily shuffling. A ByteDataset is specified by 
a `idxs.pkl` file and a `data.pkl` file.

The first 4 bytes of `idxs.pkl` is size of the dataset (number of samples),
i.e. maximum number of samples = 2^32 - 1 = 4,294,967,295
Each sample is stored in a variable-length number of bytes in `data.pkl`.
The position of sample i-th (bytes offset) in `data.pkl` is specified in `idxs.pkl`,
i.e. pos = 4 + i * idx_record_size,
where `idx_record_size` is number of bytes used to specify position of a sample, meaning 
that maximum size of `data.pkl` is about 2^48 bytes = 64 TiB.
"""

import os
import pickle
import logging
from typing import Text

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ByteDataset(Dataset):
    def __init__(
        self,
        data_path: Text,
        idx_record_size: int = 6,
        transform=None
    ):
        self.data_path = data_path
        self.idx_reader = open(os.path.join(data_path, "idxs.pkl"), "rb")
        self.data_reader = open(os.path.join(data_path, "data.pkl"), "rb")
        self.idx_record_size = idx_record_size
        self.transform = transform
    
    def __len__(self):
        self.idx_reader.seek(0, 0)
        dataset_size = self.idx_reader.read(4)
        dataset_size = int.from_bytes(dataset_size, byteorder='big', signed=False)
        return dataset_size
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop or len(self)
            step = idx.step or 1
            idxs = range(start, stop, step)
            return [self[i] for i in idxs]

        if idx >= len(self):
            raise StopIteration

        # get position of record
        self.idx_reader.seek(idx * self.idx_record_size + 4, 0)
        position = self.idx_reader.read(self.idx_record_size)
        position = int.from_bytes(position, 'big', signed=False)

        # get record
        self.data_reader.seek(position, 0)
        try:
            record = pickle.load(self.data_reader)
        except Exception as e:
            print("Idx: {} - Position: {}".format(idx, position))
            raise

        # transform
        if self.transform:
            return self.transform(record)
        return record

    def close(self):
        if not self.idx_reader.closed:
            self.idx_reader.close()
            logger.info("Idxs file closed")
        if not self.data_reader.closed:
            self.data_reader.close()
            logger.info("Data file closed")

    __del__ = close
