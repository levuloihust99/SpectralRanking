import os
import sys
import time
import random
import signal
import logging
import multiprocessing as mp

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


class DataGateway:
    def __init__(self, config: DataGatewayConfig):
        self.config = config
        self.data_queue = mp.Queue(maxsize=self.config.prefetch_factor)
        self.feeding_worker = mp.Process(target=self.feeding, daemon=True)
        self.current_pipeline_idx = -1
        self.pipeline_sequence = self.build_pipeline_sequence()
        self.pipelines = self.build_pipelines()

    def start_worker(self):
        self.feeding_worker.start()

    def build_pipeline_sequence(self):
        sequence = []
        for k, v in self.config.regulate_factors.items():
            sequence.extend([k] * v)
        return sequence

    def build_pipelines(self):
        pipelines = {}
        rnd = random.Random(self.config.seed)  # RNG set
        for pipeline in self.config.regulate_factors:
            pipelines[pipeline] = PIPELINE_CONSTRUCTOR_MAP[pipeline](
                **self.config.pipelines[pipeline].model_dump(exclude={"type"}),
                seed=rnd.getrandbits(128)  # RNG hit
            )
        return pipelines

    def __iter__(self):
        return self

    @timer(task_name="Produce batch")
    def __next__(self):
        batch = []
        while len(batch) < self.config.batch_size:
            batch.append(self.data_queue.get())
        return batch

    def collate_fn(self, items):
        return items

    def feeding(self):
        logger.info("Feeding worker started")

        while True:
            self.current_pipeline_idx = (self.current_pipeline_idx + 1) % len(
                self.pipeline_sequence
            )
            pipeline_name = self.pipeline_sequence[self.current_pipeline_idx]
            pipeline = self.pipelines[pipeline_name]
            items = next(pipeline)
            for item in items:
                self.data_queue.put(item)
