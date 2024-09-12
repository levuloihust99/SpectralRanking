import os
import sys
import time
import queue
import signal
import random
import logging
import multiprocessing as mp
from typing import Union

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
    def __init__(self, config: DataGatewayConfig):
        self.config = config
        self.data_queue = mp.Queue(maxsize=self.config.prefetch_factor)
        self.feeding_worker = None
        self.current_pipeline_idx = -1
        self.pipeline_sequence = self.build_pipeline_sequence()
        self.pipelines = self.build_pipelines()
        self.state = {}
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
                seed=rnd.getrandbits(128)  # RNG hit
            )
        return pipelines

    def set_state(self, data_gateway_state: dict):
        for pipeline_type, pipeline_state in data_gateway_state.items():
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

    def update_state(self, pipeline_type: PipelineType, state: dict):
        self.state[pipeline_type] = state

    @timer(task_name="Produce batch")
    def __next__(self):
        batch = []
        while len(batch) < self.config.batch_size:
            item = self.data_queue.get()
            self.update_state(item["type"], item["state"])
            batch.append(item)
        return batch

    def collate_fn(self, items):
        return items

    def feeding(self):
        logger.info("Feeding worker started")

        def signal_handler(signalnum, stackframe):
            logger.info("Feeding worker shutdown")
            sys.exit("Shutdown worker")

        signal.signal(signal.SIGTERM, signal_handler)

        while True:
            self.current_pipeline_idx = (self.current_pipeline_idx + 1) % len(
                self.pipeline_sequence
            )
            pipeline_name = self.pipeline_sequence[self.current_pipeline_idx]
            pipeline = self.pipelines[pipeline_name]
            items = next(pipeline)
            for item in items:
                self.data_queue.put(item)
