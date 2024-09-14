import copy
import torch
import logging
from collections import defaultdict
from typing import Optional, Union

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer

from lib.nn.modeling import T5CrossEncoder
from lib.utils.dist_utils import all_gather_list
from lib.utils.reporters import WandBReporter
from lib.v2.data_helpers.eval_dataloader import EvalDataLoader
from lib.v2.data_helpers.gateway import DataGateway
from lib.v2.nn.configuration import CrossEncoderConfig
from lib.v2.nn.ctx import distributed_context

logger = logging.getLogger(__name__)


def equal_chunking(items: list, n_chunks: int):
    L = len(items)
    init_chunk_size = L // n_chunks
    remainder = L - init_chunk_size * n_chunks
    added = [1] * remainder + [0] * (n_chunks - remainder)
    chunk_sizes = [init_chunk_size + added[i] for i in range(L)]
    idx = 0
    chunks = []
    for chunk_size in chunk_sizes:
        chunks.append(items[idx : idx + chunk_size])
        idx += chunk_size
    return chunks


class CrossEncoderTrainer:
    def __init__(
        self,
        model: Union[T5CrossEncoder, DDP],
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data_config.tokenizer_path
        )
        self.run_id = run_id
        self.reporters = []
        for reporter_name in self.config.report_to:
            if reporter_name == "wandb":
                self.reporters.append(
                    WandBReporter(api_key=self.config.wandb_api_key, run_id=self.run_id)
                )
        self.trainer_state = {}

    def checkpoint_state(self):
        return copy.deepcopy(self.trainer_state)

    def fetch_batch(self):
        batch = next(self.train_dataloader)
        self.trainer_state["data_state"] = self.train_dataloader.checkpoint_state()

        # build contrastive samples
        contrastive_samples = defaultdict(list)
        tracker = defaultdict(set)
        for item in batch["items"]:
            if (
                item["negative"]["unique_id"]
                not in tracker[item["positive"]["unique_id"]]
            ):
                contrastive_samples[item["positive"]["unique_id"]].append(
                    item["negative"]["unique_id"]
                )
            tracker[item["positive"]["unique_id"]].add(item["negative"]["unique_id"])

        # chunk batch
        lookup = batch["lookup"]
        unique_ids_and_tokens_ids = []
        for k, v in lookup.items():
            unique_ids_and_tokens_ids.append((k, v["concat_tokens_ids"]))
        if distributed_context["world_size"] > 1:
            chunked_unique_ids_and_tokens_ids = equal_chunking(
                unique_ids_and_tokens_ids, n_chunks=distributed_context["world_size"]
            )[distributed_context["local_rank"]]
        else:
            chunked_unique_ids_and_tokens_ids = unique_ids_and_tokens_ids

        # build tensors
        unique_ids = []
        batch_input_ids = []
        max_len = -1
        for unique_id, tokens_ids in chunked_unique_ids_and_tokens_ids:
            unique_ids.append(unique_id)
            batch_input_ids.append(tokens_ids)
            if max_len < len(tokens_ids):
                max_len = len(tokens_ids)

        padded_batch_input_ids = []
        batch_attention_masks = []
        for input_ids in batch_input_ids:
            pad_len = max_len - len(input_ids)
            attn_mask = [1] * len(input_ids) + [0] * pad_len
            batch_attention_masks.append(attn_mask)
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_batch_input_ids.append(padded_input_ids)
        padded_batch_input_ids = torch.tensor(
            padded_batch_input_ids, device=distributed_context["device"]
        )
        batch_attention_masks = torch.tensor(
            batch_attention_masks, device=distributed_context["device"]
        )

        return {
            "unique_ids": unique_ids,
            "input_ids": padded_batch_input_ids,
            "attention_mask": batch_attention_masks,
            "contrastive_samples": contrastive_samples,
        }

    def train(self):
        self.model.train()
        logger.info("*************** Start training ***************")
        for reporter in self.reporters:
            reporter.init_run(
                project_name="SpectralRankingV2", config=self.config.model_dump()
            )

        trained_steps = 0
        with self.train_dataloader.pipeline_context():
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
        self.optimizer.zero_grad()
        scores = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )  # [bsz]
        local_score_map = {}
        for i, unique_id in enumerate(batch["unique_ids"]):
            local_score_map[unique_id] = scores[i]

        if distributed_context["world_size"] > 1:
            all_local_score_maps = all_gather_list(local_score_map)
            global_score_map = {}
            for score_map in all_local_score_maps:
                for k, v in score_map:
                    if k in local_score_map:
                        global_score_map[k] = local_score_map[k]
                    else:
                        global_score_map[k] = v
        else:
            global_score_map = local_score_map

        step_loss = []
        contrastive_samples = batch["contrastive_samples"]
        for positive_id, negatives_ids in contrastive_samples.items():
            pos_score = global_score_map[positive_id]
            negs_scores = [global_score_map[neg_id] for neg_id in negatives_ids]
            pos_and_negs_score = torch.stack(([pos_score] + negs_scores))
            logits = F.log_softmax(pos_and_negs_score, dim=-1)
            step_loss.append(logits[0] * -1)
        step_loss = torch.stack(step_loss)
        step_loss = torch.mean(step_loss)

        scalar_step_loss = step_loss.item()
        _step_loss = (
            step_loss / distributed_context["world_size"] * self.config.score_scale
        )
        _step_loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.scheduler.step()

        return scalar_step_loss
