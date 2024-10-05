import hashlib
from pydantic import BaseModel

from lib.data_helpers.bytedataset import ByteDataset
from lib.v2.data_helpers.pipelines.idxs import IdxsGenerator
from lib.v2.data_helpers.schemas import PipelineType

from .base import BasePipelineState, BasePipeline


class NLIStratifiedFetchState(BaseModel):
    epoch: int
    iteration: int


class NLIStratifiedPipelineState(BaseModel, BasePipelineState):
    fetch_state: NLIStratifiedFetchState
    consumed: int
    buffer_index: int


class NLIStratifiedPipeline(BasePipeline):
    def __init__(
        self,
        dataset_path: str,
        buffer_size: int,
        seed: int,
    ):
        super(NLIStratifiedPipeline, self).__init__(buffer_size, seed)
        self.dataset_path = dataset_path
        self.dataset = ByteDataset(data_path=self.dataset_path)
        self.idxs_generator = IdxsGenerator(num=len(self.dataset), seed=self.seed)

    def get_fetch_state(self):
        return {
            "epoch": self.idxs_generator.epoch,
            "iteration": self.idxs_generator.iteration,
        }

    def fetch(self):
        # save fetch state for reproduction
        fetch_state = self.get_fetch_state()

        idx = next(self.idxs_generator)
        sample = self.dataset[idx]
        hypothesis_types = ["entailment", "neutral", "contradiction"]
        L = len(hypothesis_types)
        comparison_types = []
        for i in range(L - 1):
            for j in range(i + 1, L):
                comparison_types.append(
                    (hypothesis_types[i], hypothesis_types[j])
                )
        items = []
        for positive_type, negative_type in comparison_types:
            positives = sample[positive_type]
            negatives = sample[negative_type]
            for positive in positives.values():
                unique_positive_id = hashlib.md5(
                    "{}\u2581{}".format(sample["anchor"], positive["content"]).encode(
                        "utf-8"
                    )
                ).hexdigest()
                positive["unique_positive_id"] = unique_positive_id
            for negative in negatives.values():
                unique_negative_id = hashlib.md5(
                    "{}\u2581{}".format(sample["anchor"], negative["content"]).encode(
                        "utf-8"
                    )
                ).hexdigest()
                negative["unique_negative_id"] = unique_negative_id
            for positive in positives.values():
                for negative in negatives.values():
                    items.append(
                        {
                            "input": sample["anchor"],
                            "positive": {
                                "content": positive["content"],
                                "unique_id": positive["unique_positive_id"],
                                "type": positive_type,
                            },
                            "negative": {
                                "content": negative["content"],
                                "unique_id": negative["unique_negative_id"],
                                "type": negative_type,
                            },
                        }
                    )

        for i, item in enumerate(items):
            item.update(state={"fetch_state": fetch_state.copy(), "consumed": i + 1})
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            item["state"].update(buffer_index=self.buffer_index)
            item["type"] = PipelineType.NLI_STRATIFIED
            self.buffer.append(item)

    def set_state(self, state: NLIStratifiedPipelineState):
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
