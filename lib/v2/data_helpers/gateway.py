import os
import sys
import time
import queue
import signal
import random
import logging
import multiprocessing as mp
from collections import deque
from typing import Optional, Union

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
    SlidingPipeline,
    NonconcisenessPipeline,
    NoncoherencePipeline,
    NoncoveragePipeline,
    ModelComparePipeline,
    NegPoolPipeline,
]


class DataGateway:
    def __init__(
        self,
        config: DataGatewayConfig,
        tokenizer=None,
    ):
        self.config = config
        self.data_queue = mp.Queue(maxsize=self.config.prefetch_factor)
        self.feeding_worker = None
        self.current_pipeline_idx = -1
        self.pipeline_sequence = self.build_pipeline_sequence()
        self.pipelines = self.build_pipelines()
        self.state = {
            "pipelines_state": {},
            "current_pipeline_idx": self.current_pipeline_idx,
        }
        self.tokenizer = tokenizer
        self.worker_pid = None

    def start_worker(self):
        self.feeding_worker = mp.Process(target=self.feeding, daemon=True)
        self.feeding_worker.start()
        self.worker_pid = self.feeding_worker.pid

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

    def set_state(self, data_gateway_state: dict):
        self.current_pipeline_idx = data_gateway_state["current_pipeline_idx"]
        pipelines_state = data_gateway_state["pipelines_state"]
        for pipeline_type, pipeline_state in pipelines_state.items():
            self.pipelines[pipeline_type].set_state(
                PIPELINE_STATE_MAP[pipeline_type](**pipeline_state)
            )
            self.pipelines[pipeline_type].close_dataset()
        if self.worker_pid:
            os.kill(self.worker_pid, signal.SIGTERM)
            self.clear_queue()
        self.start_worker()

    @timer(task_name="Clear queue")
    def clear_queue(self):
        logger.info("Clear data queue...")
        t0 = time.perf_counter()
        MAX_WAIT = 10
        while True:
            try:
                if time.perf_counter() - t0 > MAX_WAIT:
                    raise Exception("Queue is not cleared within {}s".format(MAX_WAIT))
                self.data_queue.get_nowait()
            except queue.Empty as exc:
                break

    def __iter__(self):
        return self

    def update_state(self, pipeline_type: PipelineType, state: dict, pipeline_idx: int):
        self.state["pipelines_state"][pipeline_type] = state
        self.state["current_pipeline_idx"] = pipeline_idx

    @timer(task_name="Produce batch")
    def __next__(self):
        batch = self.data_queue.get()
        for item, pipeline_idx in batch:
            self.update_state(item["type"], item["state"], pipeline_idx)
        return batch

    def collate_fn(self, items):
        if not self.tokenizer:
            return items

        input_texts = []
        for item, _ in items:
            input_texts.append(item["input"])

        tokenize_input_kwargs = {}
        if self.config.max_input_len:
            tokenize_input_kwargs["max_length"] = self.config.max_input_len
            tokenize_input_kwargs["truncation"] = True

        input_texts_token_ids = self.tokenizer(
            input_texts,
            padding=False,
            return_attention_mask=False,
            **tokenize_input_kwargs,
        )

        lookup = {}
        tokenize_output_kwargs = {}
        if self.config.max_output_len:
            tokenize_output_kwargs["max_length"] = self.config.max_output_len
            tokenize_output_kwargs["truncation"] = True

        for item, _ in items:
            if item["positive"]["unique_id"] not in lookup:
                tokens_ids = self.tokenizer(
                    item["positive"]["content"],
                    padding=False,
                    return_attention_mask=False,
                    **tokenize_output_kwargs,
                )
                lookup[item["positive"]["unique_id"]] = {
                    **item["positive"],
                    "tokens_ids": tokens_ids,
                }
            else:
                assert (
                    lookup[item["positive"]["unique_id"]]["content"]
                    == item["positive"]["content"]
                )
            if item["negative"]["unique_id"] not in lookup:
                tokens_ids = self.tokenizer(
                    item["negative"]["content"],
                    padding=False,
                    return_attention_mask=False,
                    **tokenize_output_kwargs,
                )
                lookup[item["negative"]["unique_id"]] = {
                    **item["negative"],
                    "tokens_ids": tokens_ids,
                }
            else:
                assert (
                    lookup[item["negative"]["unique_id"]]["content"]
                    == item["negative"]["content"]
                )

    def feeding(self):
        logger.info("Feeding worker started")

        def signal_handler(signalnum, stackframe):
            logger.info("Feeding worker shutdown (signal {})".format(signalnum))
            for pipeline in self.pipelines.values():
                pipeline.close_dataset()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)

        buffer = deque(maxlen=10000)
        while True:
            saved_pipeline_idx = self.current_pipeline_idx
            self.current_pipeline_idx = (self.current_pipeline_idx + 1) % len(
                self.pipeline_sequence
            )
            pipeline_name = self.pipeline_sequence[self.current_pipeline_idx]
            pipeline = self.pipelines[pipeline_name]
            items = next(pipeline)
            items_with_meta = []
            for i, item in enumerate(items):
                if i < len(items) - 1:
                    items_with_meta.append((item, saved_pipeline_idx))
                else:
                    items_with_meta.append((item, self.current_pipeline_idx))
            buffer.extend(items_with_meta)
            if len(buffer) > self.config.batch_size:
                batch = []
                for _ in range(self.config.batch_size):
                    batch.append(buffer.popleft())
                transformed_batch = self.collate_fn(batch)
                self.data_queue.put(transformed_batch)

    def __del__(self):
        if self.worker_pid:
            os.kill(self.worker_pid, signal.SIGTERM)
        logger.info("Data gateway closed")
