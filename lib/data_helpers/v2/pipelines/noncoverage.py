import uuid
from collections import deque
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from .idxs import IdxsGenerator


class NoncoverageState(BaseModel):
    epoch: int
    iteration: int
    index: int
    is_last: bool


class NoncoveragePipeline:
    """
    Attributes:

    """

    def __init__(self, dataset_path: str, buffer_size: int, seed: int):
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(data_path=dataset_path)
        self.buffer_size = buffer_size
        self.seed = seed
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
            if comp["metadata"]["type"] == "Non-coverage":
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

        self.buffer.extend(items)
        
    def fill_buffer(self):
        """Fill buffer with items that are consumed by the data gateway."""
        
        while len(self.buffer) < self.buffer_size:
            self.fetch()

    def __next__(self):
        # fill buffer
        while len(self.buffer) < self.buffer_size:
            self.fetch()

        # fetch from buffer
        fetched_items = []
        for _ in range(self.bsz):
            fetched_items.append(self.buffer.popleft())

        return fetched_items

    def set_state(self, state: NoncoverageState):
        """Recover the state of the pipeline. Check if the last consumed data indicates
        the completion of a batch. If True, set state to using `epoch` and `iteration`.
        Otherwise, step back to reproduce the batch and ignore consumed items."""

        self.buffer.clear()
        self.idxs_generator.set_state(epoch=state.epoch, iteration=state.iteration)
        if state.is_last is False:
            self.idxs_generator.step_back()
            next(self)  # reproduce the non-complete batch
            # ignore consumed items
            for item in self.buffer:
                if item["state"]["index"] <= state.index:
                    self.buffer.popleft()
