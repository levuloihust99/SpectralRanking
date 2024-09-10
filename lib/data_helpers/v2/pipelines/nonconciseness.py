import uuid
from collections import deque
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from .idxs import IdxsGenerator


class NonconcisenessState(BaseModel):
    epoch: int
    iteration: int
    index: int
    is_last: bool


class NonconcisenessPipeline:
    """
    Attributes:
        dataset_path (str): path to the byte dataset
        bsz (int): batch size, the number of items that __next__ produces
        seed (int): for reproducible
        buffer (deque): items are fetched from buffer
        idxs_generator (IdxsGenerator): generate idxs used to access actual data
    """

    def __init__(
        self,
        dataset_path: str,
        bsz: int,
        seed: int,
    ):
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(data_path=dataset_path)
        self.seed = seed
        self.bsz = bsz
        self.buffer = deque()
        self.idxs_generator = IdxsGenerator(num=len(self.dataset), seed=seed)

    def __iter__(self):
        return self

    def fetch(self):
        """Fetch items from next sample. One sample from the dataset can map to multiple items."""

        idx = next(self.idxs_generator)
        sample = self.dataset[idx]
        items = []
        for comp in sample["comparisons"]:
            if comp["metadata"]["type"] == "Non-conciseness":
                for positive in comp["positives"]:
                    positive["unique_id"] = uuid.uuid4().hex
                for negative in comp["negatives"]:
                    negative["unique_id"] = uuid.uuid4().hex
                for positive in comp["positives"]:
                    for negative in comp["negatives"]:
                        items.append(
                            {
                                "positive": {
                                    "content": positive["content"],
                                    "unique_id": positive["unique_id"],
                                },
                                "negative": {
                                    "content": negative["content"],
                                    "unique_id": negative["unique_id"],
                                },
                            }
                        )
        for i, item in enumerate(items):
            item.update(
                state={
                    "epoch": self.idxs_generator.epoch,
                    "iteration": self.idxs_generator.iteration,
                    "index": i,
                    "is_last": i == len(items) - 1,
                }
            )

        return items

    def __next__(self):
        # fill buffer
        while len(self.buffer) < self.bsz:
            items = self.fetch()
            self.buffer.extend(items)

        # fetch from buffer
        fetched_items = []
        for _ in range(self.bsz):
            fetched_items.append(self.buffer.popleft())

        return fetched_items

    def set_state(self, state: NonconcisenessState):
        self.buffer.clear()
        self.idxs_generator.set_state(epoch=state.epoch, iteration=state.iteration)
        if state.is_last is False:
            self.idxs_generator.step_back()
            items = self.fetch()
            for i, item in enumerate(items):
                if i > state.index:
                    self.buffer.append(item)
