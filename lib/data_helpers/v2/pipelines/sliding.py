import random
from collections import deque
from typing import Literal
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from ..id import get_unique_id
from .idxs import IdxsGenerator
from ..schemas import PipelineType


class SlidingFetchState(BaseModel):
    epoch: int
    iteration: int
    sliding_buf_size: int


class SlidingPipelineState(BaseModel):
    fetch_state: SlidingFetchState
    consumed: int
    buffer_index: int


class SlidingPipeline:
    def __init__(
        self, dataset_path: str, buffer_size: int, contrastive_size: int, seed: int
    ):
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(dataset_path)
        self.buffer_size = buffer_size
        self.prev_fetch_state = None
        self.contrastive_size = contrastive_size
        self.seed = seed
        self.buffer = deque()
        self.buffer_index = -1
        self.shift = 0
        self.sliding_buffer = deque()
        self.idxs_generator = IdxsGenerator(num=len(self.dataset), seed=self.seed)

    def fetch_positive(self):
        rnd = random.Random(
            self.seed
            + self.idxs_generator.epoch * len(self.dataset)
            + self.idxs_generator.iteration
        )  # RNG set
        idx = next(self.idxs_generator)
        sample = self.dataset[idx]
        all_positives = []
        tracker = set()
        for comp in sample["comparisons"]:
            if comp["metadata"]["type"] in {
                "Non-conciseness",
                "Non-coverage",
                "different-models",
            }:
                for item in comp["positives"] + comp["negatives"]:
                    if item["content"] not in tracker:
                        tracker.add(item["content"])
                        all_positives.append(item)

            if comp["metadata"]["type"] in {"Non-coherence"}:
                for positive in comp["positives"]:
                    if positive["content"] not in tracker:
                        tracker.add(positive["content"])
                        all_positives.append(positive)

        item = rnd.choice(all_positives)  # RNG hit
        item["unique_id"] = get_unique_id(rnd)  # RNG hit
        item["input"] = sample["input"]
        self.sliding_buffer.append(item)

    def get_fetch_state(self):
        return {
            "epoch": self.idxs_generator.epoch,
            "iteration": self.idxs_generator.iteration,
            "sliding_buf_size": len(self.sliding_buffer),
        }

    def fetch(self):
        """Return next items to be appended to the buffer."""

        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        while len(self.sliding_buffer) < self.contrastive_size + 1:
            self.fetch_positive()

        positive = self.sliding_buffer.popleft()
        negatives = list(self.sliding_buffer)
        items = []
        for i, negative in enumerate(negatives):
            items.append(
                {
                    "input": positive["input"],
                    "positive": {
                        "content": positive["content"],
                        "unique_id": positive["unique_id"],
                    },
                    "negative": {
                        "content": negative["content"],
                        "unique_id": negative["unique_id"],
                    },
                    "state": {"fetch_state": fetch_state.copy(), "consumed": i + 1},
                }
            )

        for item in items:
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            item["state"].update(buffer_index=self.buffer_index)
            item["type"] = PipelineType.SLIDING
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

    def set_state(self, state: SlidingPipelineState):
        """Setup the pipeline at an appropriate state ready for continuous data generation."""

        self.idxs_generator.set_state(
            epoch=state.fetch_state.epoch, iteration=state.fetch_state.iteration
        )
        for _ in range(state.fetch_state.sliding_buf_size):
            self.idxs_generator.step_back()
        self.buffer.clear()
        self.sliding_buffer.clear()
        self.shift = state.buffer_index + 1
        self.buffer_index = state.buffer_index
        for _ in range(state.consumed):
            self.buffer_index -= 1
            if self.buffer_index < 0:
                self.buffer_index += self.buffer_size
        self.fetch()
        for _ in range(state.consumed):
            self.buffer.popleft()
