import random
import hashlib
import itertools
from collections import deque
from typing import Literal
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset
from lib.v2.data_helpers.pipelines.idxs import IdxsGenerator
from lib.v2.data_helpers.schemas import PipelineType

from .base import BasePipelineState, BasePipeline


class NLISlidingFetchState(BaseModel):
    epoch: int
    iteration: int
    sliding_buf_size: int


class NLISlidingPipelineState(BaseModel, BasePipelineState):
    fetch_state: NLISlidingFetchState
    consumed: int
    buffer_index: int


class NLISlidingPipeline(BasePipeline):
    def __init__(
        self, dataset_path: str, contrastive_size: int, buffer_size: int, seed: int
    ):
        super(NLISlidingPipeline, self).__init__(buffer_size, seed)
        self.contrastive_size = contrastive_size
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(data_path=self.dataset_path)
        self.idxs_generator = IdxsGenerator(num=len(self.dataset), seed=self.seed)
        self.sliding_buffer = deque()

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
        iterator = itertools.chain(
            sample["entailment"].values(),
            # sample["neutral"].values(),
            # sample["contradiction"].values(),
        )
        hypothesis_types = (
            ["entailment"] * len(sample["entailment"])
            # + ["neutral"] * len(sample["neutral"])
            # + ["contradiction"] * len(sample["contradiction"])
        )
        for item, type in zip(iterator, hypothesis_types):
            if item["id"] not in tracker:
                all_positives.append({**item, "type": type})
            tracker.add(item["id"])
        picked_item = rnd.choice(all_positives)  # RNG hit
        self.sliding_buffer.append(
            {
                "input": sample["anchor"],
                "content": picked_item["content"],
                "type": picked_item["type"],
            }
        )

    def get_fetch_state(self):
        return {
            "epoch": self.idxs_generator.epoch,
            "iteration": self.idxs_generator.iteration,
            "sliding_buf_size": len(self.sliding_buffer),
        }

    def fetch(self):
        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        while len(self.sliding_buffer) < self.contrastive_size + 1:
            self.fetch_positive()

        positive = self.sliding_buffer.popleft()
        negatives = list(self.sliding_buffer)
        items = []
        unique_positive_id = hashlib.md5(
            "{}\u2581{}".format(positive["input"], positive["content"]).encode("utf-8")
        ).hexdigest()
        for i, negative in enumerate(negatives):
            unique_negative_id = hashlib.md5(
                "{}\u2581{}".format(positive["input"], negative["content"]).encode(
                    "utf-8"
                )
            ).hexdigest()
            items.append(
                {
                    "input": positive["input"],
                    "positive": {
                        "content": positive["content"],
                        "unique_id": unique_positive_id,
                        "type": positive["type"],
                    },
                    "negative": {
                        "content": negative["content"],
                        "unique_id": unique_negative_id,
                        "type": negative["type"],
                    },
                    "state": {"fetch_state": fetch_state.copy(), "consumed": i + 1},
                }
            )

        for item in items:
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            item["state"].update(buffer_index=self.buffer_index)
            item["type"] = PipelineType.NLI_SLIDING
            self.buffer.append(item)

    def set_state(self, state: NLISlidingPipelineState):
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

    def close_dataset(self):
        if self.dataset:
            self.dataset = None
