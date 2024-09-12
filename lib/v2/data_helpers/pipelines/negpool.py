import random
from collections import deque
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset

from ..id import get_unique_id
from .idxs import IdxsGenerator
from ..schemas import PipelineType


class IdxsGeneratorState(BaseModel):
    epoch: int
    iteration: int


class NegPoolFetchState(BaseModel):
    pos: IdxsGeneratorState
    neg: IdxsGeneratorState


class NegPoolPipelineState(BaseModel):
    fetch_state: NegPoolFetchState
    consumed: int
    buffer_index: int


class NegPoolConfig(BaseModel):
    pos_dataset_path: str
    neg_dataset_path: str
    buffer_size: int
    contrastive_size: int


class NegPoolPipeline:
    def __init__(
        self,
        pos_dataset_path: str,
        neg_dataset_path: str,
        buffer_size: int,
        contrastive_size: int,
        seed: int,
    ):
        self.pos_dataset_path = pos_dataset_path
        self.pos_dataset = None
        self.neg_dataset_path = neg_dataset_path
        self.neg_dataset = None
        self.buffer_size = buffer_size
        self.seed = seed
        self.buffer = deque()
        self.buffer_index = -1
        self.contrastive_size = contrastive_size
        self.shift = 0
        rnd = random.Random(seed)
        self.pos_seed = rnd.getrandbits(128)
        self.neg_seed = rnd.getrandbits(128)
        self.pos_idxs_generator = self.get_pos_idxs_generator()
        self.neg_idxs_generator = self.get_neg_idxs_generator()

    def get_pos_idxs_generator(self):
        dataset = ByteDataset(data_path=self.pos_dataset_path)
        return IdxsGenerator(num=len(dataset), seed=self.pos_seed)

    def get_neg_idxs_generator(self):
        dataset = ByteDataset(data_path=self.neg_dataset_path)
        return IdxsGenerator(num=len(dataset), seed=self.neg_seed)

    def __iter__(self):
        return self

    def get_fetch_state(self):
        return {
            "pos": {
                "epoch": self.pos_idxs_generator.epoch,
                "iteration": self.pos_idxs_generator.iteration,
            },
            "neg": {
                "epoch": self.neg_idxs_generator.epoch,
                "iteration": self.neg_idxs_generator.iteration,
            },
        }

    def fetch(self):
        """Fetch items from next sample. One sample from the dataset can map to multiple items."""

        if not self.pos_dataset:
            self.pos_dataset = ByteDataset(data_path=self.pos_dataset_path)
        if not self.neg_dataset:
            self.neg_dataset = ByteDataset(data_path=self.neg_dataset_path)

        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        # < depend on fetch state
        pos_idx = next(self.pos_idxs_generator)
        pos_sample = self.pos_dataset[pos_idx]
        neg_samples = []
        for _ in range(self.contrastive_size):
            neg_samples.append(self.neg_dataset[next(self.neg_idxs_generator)])
        # depend on fetch state />

        rnd = random.Random(
            self.seed
            + self.pos_idxs_generator.epoch * len(self.pos_dataset)
            + self.pos_idxs_generator.iteration
            + self.neg_idxs_generator.epoch * len(self.neg_dataset)
            + self.neg_idxs_generator.iteration
        )  # RNG set

        all_positives = []
        tracker = set()
        for comp in pos_sample["comparisons"]:
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

        pos_item = rnd.choice(all_positives)  # RNG hit
        pos_item["unique_id"] = get_unique_id(rnd)
        items = []
        for i, neg_sample in enumerate(neg_samples):
            items.append(
                {
                    "input": pos_sample["input"],
                    "positive": {
                        "content": pos_item["content"],
                        "unique_id": pos_item["unique_id"],  # RNG hit
                    },
                    "negative": {
                        "content": neg_sample.get("completion") or neg_sample["output"],
                        "unique_id": get_unique_id(rnd),  # RNG hit
                    },
                    "state": {"fetch_state": fetch_state.copy(), "consumed": i + 1},
                }
            )

        for item in items:
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            item["state"].update(buffer_index=self.buffer_index)
            item["type"] = PipelineType.NEGPOOL
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

    def set_state(self, state: NegPoolPipelineState):
        """Setup the pipeline at an appropriate state ready for continuous data generation."""

        self.pos_idxs_generator.set_state(
            epoch=state.fetch_state.pos.epoch, iteration=state.fetch_state.pos.iteration
        )
        self.neg_idxs_generator.set_state(
            epoch=state.fetch_state.neg.epoch, iteration=state.fetch_state.neg.iteration
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
        if self.pos_dataset:
            self.pos_dataset = None
        if self.neg_dataset:
            self.neg_dataset = None
