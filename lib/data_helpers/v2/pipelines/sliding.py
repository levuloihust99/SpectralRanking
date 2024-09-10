import random
import uuid
from collections import deque
from contextlib import contextmanager
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from ..id import get_unique_id
from .idxs import IdxsGenerator


class FetchState(BaseModel):
    epoch: int
    iteration: int


class SlidingPipelineState(BaseModel):
    fetch_state: FetchState
    consumed: int


class SlidingWithPosPipeline:
    def __init__(self, dataset_path: str, buffer_size: int, bsz: int, seed: int):
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(dataset_path)
        self.buffer_size = buffer_size
        self.prev_fetch_state = None
        self.bsz = bsz
        self.seed = seed
        self.buffer = deque()
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
            if comp["metadata"]["type"] in {"Non-conciseness", "Non-coherence"}:
                for positive in comp["positives"]:
                    if positive["content"] not in tracker:
                        tracker.add(positive["content"])
                        all_positives.append(positive)

        rnd.shuffle(all_positives)  # RNG hit
        item = all_positives[0]
        item["unique_id"] = get_unique_id(rnd)  # RNG hit
        self.sliding_buffer.append(item)

    def get_fetch_state(self):
        return {
            "epoch": self.idxs_generator.epoch,
            "iteration": self.idxs_generator.iteration,
        }

    def fetch(self):
        """Return next items to be appended to the buffer."""

        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        while len(self.sliding_buffer) < self.bsz + 1:
            self.fetch_positive()

        positive = self.sliding_buffer.popleft()
        negatives = list(self.sliding_buffer)
        items = []
        for i, negative in enumerate(negatives):
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
                    "state": {"fetch_state": fetch_state.copy(), "consumed": i + 1},
                }
            )

        self.buffer.extend(items)

    def fill_buffer(self):
        """Fill buffer with items that are consumed by the data gateway."""

        while len(self.buffer) < self.buffer_size:
            self.fetch()

    def __next__(self):
        self.fill_buffer()
        items = []
        for _ in range(self.buffer_size - self.shift):
            items.append(self.buffer.popleft())
        for idx, item in enumerate(items):
            item.update(buffer_index=idx + self.shift)
        return items

    def set_state(self, state: SlidingPipelineState):
        """Setup the pipeline at an appropriate state ready for continuous data generation."""

        self.idxs_generator.set_state(
            epoch=state.fetch_state.epoch, iteration=state.fetch_state.iteration
        )
        self.buffer.clear()
        self.fetch()
        for _ in range(state.consumed):
            self.buffer.popleft()

    @contextmanager
    def buffer_shift(self, shift: int):
        self.shift = shift
        yield
        self.shift = 0
