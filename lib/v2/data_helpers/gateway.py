from __future__ import annotations

import os
import sys
import time
import copy
import queue
import signal
import random
import logging
import multiprocessing as mp
from contextlib import contextmanager
from collections import defaultdict, deque
from typing import Optional, Union

from transformers import AutoTokenizer

from lib.utils.timing import timer

from .pipelines.sliding import SlidingPipeline, SlidingPipelineState
from .pipelines.nonconciseness import (
    NonconcisenessPipeline,
    NonconcisenessPipelineState,
)
from .pipelines.noncoherence import NoncoherencePipeline, NoncoherencePipelineState
from .pipelines.noncoverage import NoncoveragePipeline, NoncoveragePipelineState
from .pipelines.model_compare import ModelComparePipeline, ModelComparePipelineState
from .pipelines.negpool import NegPoolPipeline, NegPoolPipelineState

from .schemas import PipelineType, DataGatewayConfig

logger = logging.getLogger(__name__)


PIPELINE_CONSTRUCTOR_MAP = {
    PipelineType.SLIDING: SlidingPipeline,
    PipelineType.NON_CONCISENESS: NonconcisenessPipeline,
    PipelineType.NON_COHERENCE: NoncoherencePipeline,
    PipelineType.NON_COVERAGE: NoncoveragePipeline,
    PipelineType.MODEL_COMPARE: ModelComparePipeline,
    PipelineType.NEGPOOL: NegPoolPipeline,
}


PIPELINE_STATE_MAP = {
    PipelineType.SLIDING: SlidingPipelineState,
    PipelineType.NON_CONCISENESS: NonconcisenessPipelineState,
    PipelineType.NON_COHERENCE: NoncoherencePipelineState,
    PipelineType.NON_COVERAGE: NoncoveragePipelineState,
    PipelineType.MODEL_COMPARE: ModelComparePipelineState,
    PipelineType.NEGPOOL: NegPoolPipelineState,
}

PipelineUnion = Union[
    NoncoveragePipeline,
    ModelComparePipeline,
    NegPoolPipeline,
]


class DataGatewayWorker:
    """Used by DataGateway, suppose to run in child process."""

    def __init__(
        self,
        config: DataGatewayConfig,
        queue: mp.Queue,
        stop_event,
        state: Optional[dict] = None,
    ):
        self.config = config
        self.queue = queue
        self.stop_event = stop_event
        self.current_pipeline_idx = -1
        self.pipeline_sequence = self.build_pipeline_sequence()
        self.pipelines = self.build_pipelines()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if state:
            self.set_state(state)

    def build_pipeline_sequence(self):
        sequence = []
        for k, v in self.config.regulate_factors.items():
            sequence.extend([k] * v)
        return sequence

    def build_pipelines(self) -> dict[PipelineType, PipelineUnion]:
        pipelines = {}
        rnd = random.Random(self.config.seed)  # RNG set
        for pipeline in self.config.regulate_factors:
            pipelines[pipeline] = PIPELINE_CONSTRUCTOR_MAP[pipeline](
                **self.config.pipelines[pipeline].model_dump(exclude={"type"}),
                seed=rnd.getrandbits(128),  # RNG hit
            )
        return pipelines

    def set_state(self, state: dict):
        self.current_pipeline_idx = state["current_pipeline_idx"]
        pipelines_state = state["pipelines_state"]
        for pipeline_type, pipeline_state in pipelines_state.items():
            self.pipelines[pipeline_type].set_state(
                PIPELINE_STATE_MAP[pipeline_type](**pipeline_state)
            )

    def collate_fn(self, items):
        lookup = {}
        for item, _ in items:
            # < document
            document_tokens = self.tokenizer.tokenize(item["input"])
            if self.config.max_input_len:
                document_tokens = document_tokens[: self.config.max_input_len]
            # document />

            # < positive
            if item["positive"]["unique_id"] not in lookup:
                summary_tokens = self.tokenizer.tokenize(item["positive"]["content"])
                if self.config.max_output_len:
                    summary_tokens = summary_tokens[: self.config.max_output_len]
                concat_tokens = (
                    document_tokens
                    + [self.config.sep_token]
                    + summary_tokens
                    + [self.tokenizer.eos_token]
                )
                lookup[item["positive"]["unique_id"]] = {
                    "input": item["input"],
                    **item["positive"],
                    "concat_tokens_ids": self.tokenizer.convert_tokens_to_ids(
                        concat_tokens
                    ),
                }
            else:
                assert (
                    lookup[item["positive"]["unique_id"]]["content"]
                    == item["positive"]["content"]
                )
            # positive />

            # < negative
            if item["negative"]["unique_id"] not in lookup:
                summary_tokens = self.tokenizer.tokenize(item["negative"]["content"])
                if self.config.max_output_len:
                    summary_tokens = summary_tokens[: self.config.max_output_len]
                concat_tokens = (
                    document_tokens
                    + [self.config.sep_token]
                    + summary_tokens
                    + [self.tokenizer.eos_token]
                )
                lookup[item["negative"]["unique_id"]] = {
                    "input": item["input"],
                    **item["negative"],
                    "concat_tokens_ids": self.tokenizer.convert_tokens_to_ids(
                        concat_tokens
                    ),
                }
            else:
                assert (
                    lookup[item["negative"]["unique_id"]]["content"]
                    == item["negative"]["content"]
                )
            # negative />

        return {
            "items": items,
            "lookup": lookup,
        }

    def start(self):
        logger.info("Feeding loop starts")

        buffer = deque(maxlen=self.config.batch_size * 10)
        try:
            first_flag = True
            while not self.stop_event.is_set():
                saved_pipeline_idx = self.current_pipeline_idx
                self.current_pipeline_idx = (self.current_pipeline_idx + 1) % len(
                    self.pipeline_sequence
                )
                pipeline_name = self.pipeline_sequence[self.current_pipeline_idx]
                pipeline = self.pipelines[pipeline_name]
                items = next(pipeline)
                if not items and not first_flag:
                    items = next(pipeline)
                first_flag = False
                items_with_meta = []
                for i, item in enumerate(items):
                    items_with_meta.append((item, saved_pipeline_idx))
                buffer.extend(items_with_meta)
                if len(buffer) >= self.config.batch_size:
                    batch = []
                    for _ in range(self.config.batch_size):
                        batch.append(buffer.popleft())
                    transformed_batch = self.collate_fn(batch)
                    self.queue.put(transformed_batch)

        except Exception as e:
            logger.error("Exception in feeding worker", exc_info=True)

        finally:
            logger.info("Feeding worker exiting")


def worker_entrypoint(
    config: DataGatewayConfig, queue: mp.Queue, stop_event, state: Optional[dict] = None
):
    worker = DataGatewayWorker(
        config=config, queue=queue, stop_event=stop_event, state=state
    )
    worker.start()


class DataGateway:
    def __init__(self, config: DataGatewayConfig):
        self.config = config
        self.queue = None
        self.stop_event = None
        self.worker = None
        self.state = {
            "pipelines_state": {},
            "current_pipeline_idx": -1,
        }

    def checkpoint_state(self):
        return copy.deepcopy(self.state)

    def start_worker(self):
        self.queue = mp.Queue(maxsize=self.config.prefetch_factor)
        self.stop_event = mp.Event()
        self.worker = mp.Process(
            target=worker_entrypoint,
            args=(self.config, self.queue, self.stop_event, self.state),
        )
        self.worker.start()

    def set_state(self, state: dict):
        if self.worker is not None:
            raise Exception("Cannot set state inside pipeline context")
        self.state = state

    def __iter__(self):
        return self

    def update_state(self, pipeline_type: PipelineType, state: dict, pipeline_idx: int):
        self.state["pipelines_state"][pipeline_type] = state
        self.state["current_pipeline_idx"] = pipeline_idx

    @timer(task_name="Produce batch")
    def __next__(self):
        batch = self.queue.get()
        pure_items = []
        for item, pipeline_idx in batch["items"]:
            pure_items.append(item)
            self.update_state(item["type"], item["state"], pipeline_idx)
        return {"items": pure_items, "lookup": batch["lookup"]}

    @timer(task_name="Clean up queue")
    def cleanup_queue(self):
        MAX_WAIT = 2
        logger.info("Queue will be cleared within {}s".format(MAX_WAIT))
        while True:
            try:
                self.queue.get(timeout=MAX_WAIT)
            except queue.Empty:
                break

    def shutdown(self):
        logger.info("Shutting down DataGateway")
        self.stop_event.set()
        self.cleanup_queue()
        if self.worker:
            self.worker.join()
        self.worker = None

    @contextmanager
    def pipeline_context(self):
        try:
            self.start_worker()
            yield
        finally:
            self.shutdown()
