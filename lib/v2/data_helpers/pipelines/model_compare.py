import random
from collections import deque
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from ..id import get_unique_id
from .idxs import IdxsGenerator
from ..schemas import PipelineType


class ModelCompareFetchState(BaseModel):
    epoch: int
    iteration: int


class ModelComparePipelineState(BaseModel):
    fetch_state: ModelCompareFetchState
    consumed: int
    buffer_index: int


class ModelComparePipeline:
    def __init__(self, dataset_path: str, buffer_size: int, seed: int):
        self.dataset_path = dataset_path
        self.dataset = None
        self.buffer_size = buffer_size
        self.seed = seed
        self.buffer = deque()
        self.buffer_index = -1
        self.shift = 0
        self.idxs_generator = self.get_idxs_generator()

    def get_idxs_generator(self):
        dataset = ByteDataset(data_path=self.dataset_path)
        return IdxsGenerator(num=len(dataset), seed=self.seed)

    def __iter__(self):
        return self

    def get_fetch_state(self):
        return {
            "epoch": self.idxs_generator.epoch,
            "iteration": self.idxs_generator.iteration,
        }

    def fetch(self):
        """Fetch items from next sample. One sample from the dataset can map to multiple items."""

        if not self.dataset:
            self.dataset = ByteDataset(data_path=self.dataset_path)

        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        idx = next(self.idxs_generator)
        sample = self.dataset[idx]
        items = []
        rnd = random.Random(
            self.seed
            + self.idxs_generator.epoch * len(self.dataset)
            + self.idxs_generator.iteration
        )  # RNG set
        for comp in sample["comparisons"]:
            if comp["metadata"]["type"] == "different-models":
                for positive in comp["positives"]:
                    positive["unique_id"] = get_unique_id(rnd)
                for negative in comp["negatives"]:
                    negative["unique_id"] = get_unique_id(rnd)
                for positive in comp["positives"]:
                    for negative in comp["negatives"]:
                        items.append(
                            {
                                "input": sample["input"],
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

        rnd.shuffle(items)
        items = items[: self.buffer_size]
        for i, item in enumerate(items):
            item.update(state={"fetch_state": fetch_state, "consumed": i + 1})

        for item in items:
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            item["state"].update(buffer_index=self.buffer_index)
            item["type"] = PipelineType.MODEL_COMPARE
            self.buffer.append(item)

    def fill_buffer(self):
        """Fill buffer with items that are consumed by the data gateway."""

        while len(self.buffer) < self.buffer_size:
            self.fetch()

    def __next__(self):
        self.fill_buffer()
        items = []
        for _ in range(self.buffer_size - self.shift):
            items.append(self.buffer.popleft())
        self.shift = 0
        return items

    def set_state(self, state: ModelComparePipelineState):
        """Setup the pipeline at an appropriate state ready for continuous data generation."""

        self.idxs_generator.set_state(
            epoch=state.fetch_state.epoch, iteration=state.fetch_state.iteration
        )
        self.buffer.clear()
        self.shift = state.buffer_index + 1
        self.buffer_index = state.buffer_index
        for _ in range(state.consumed):
            self.buffer_index -= 1
            if self.buffer_index < 0:
                self.buffer_index += self.buffer_size
        self.fetch()
        for _ in range(state.consumed):
            self.buffer.popleft()

    def close_dataset(self):
        if self.dataset:
            self.dataset = None
