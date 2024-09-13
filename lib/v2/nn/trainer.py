import torch
import logging
from collections import defaultdict
from typing import Optional

from lib.nn.modeling import T5CrossEncoder
from lib.utils.reporters import WandBReporter
from lib.v2.data_helpers.eval_dataloader import EvalDataLoader
from lib.v2.data_helpers.gateway import DataGateway
from lib.v2.nn.configuration import CrossEncoderConfig
from lib.v2.nn.ctx import distributed_context

logger = logging.getLogger(__name__)


class CrossEncoderTrainer:
    def __init__(
        self,
        model: T5CrossEncoder,
        config: CrossEncoderConfig,
        train_dataloader: DataGateway,
        eval_dataloader: Optional[EvalDataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        run_id: str,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_id = run_id
        self.reporters = []
        for reporter_name in self.config.report_to:
            if reporter_name == "wandb":
                self.reporters.append(
                    WandBReporter(api_key=self.config.wandb_api_key, run_id=self.run_id)
                )
        self.trainer_state = {}

    def fetch_batch(self):
        batch = next(self.train_dataloader)
        self.trainer_state["data_state"] = self.train_dataloader.state
        distributed_world_size = distributed_context["world_size"]
        if distributed_world_size > 1:
            pass
        else:
            return batch

    def train(self):
        self.model.train()
        logger.info("*************** Start training ***************")
        for reporter in self.reporters:
            reporter.init_run(
                project_name="SpectralRankingV2", config=self.config.model_dump()
            )

        trained_steps = 0
        for step in range(trained_steps, self.config.num_train_steps):
            batch = self.fetch_batch()
            loss = self.train_step(batch)  # scalar loss value

            # state update and log at the end of the iteration
            trained_steps = step + 1
            self.trainer_state["trained_steps"] = trained_steps
            if self.config.logging_steps % trained_steps == 0:
                for reporter in self.reporters:
                    reporter.log({"train/loss": 0.0}, step=trained_steps)

    def train_step(self, batch):
        pass
