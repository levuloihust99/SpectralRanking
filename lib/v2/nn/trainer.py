import os
import time
import copy
import json
import torch
import shutil
import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Optional, Union

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer

from lib.nn.grad_cache import RandContext
from lib.nn.modeling import T5CrossEncoder
from lib.utils.dist_utils import all_gather_list
from lib.utils.reporters import WandBReporter
from lib.v2.data_helpers.eval_dataloader import EvalDataLoader
from lib.v2.data_helpers.gateway import DataGateway
from lib.v2.nn.configuration import CrossEncoderConfig
from lib.v2.nn.ctx import distributed_context

logger = logging.getLogger(__name__)


def equal_chunking(items: list, n_chunks: int, tracker: Optional[dict] = None):
    L = len(items)
    init_chunk_size = L // n_chunks
    remainder = L - init_chunk_size * n_chunks
    added = [1] * remainder + [0] * (n_chunks - remainder)
    if isinstance(tracker, dict):
        tracker["added"] = added
    chunk_sizes = [init_chunk_size + added[i] for i in range(n_chunks)]
    idx = 0
    chunks = []
    for chunk_size in chunk_sizes:
        chunks.append(items[idx : idx + chunk_size])
        idx += chunk_size
    return chunks


def equal_chunking_with_maxsize(
    items: list, max_size: int, tracker: Optional[dict] = None
):
    L = len(items)
    n_chunks = (L - 1) // max_size + 1
    return equal_chunking(items, n_chunks, tracker)


def get_model_obj(model: torch.nn.Module):
    return model.module if hasattr(model, "module") else model


def save_rng_state():
    cpu_state = torch.get_rng_state()
    if distributed_context["device"].type == "cuda":
        device_index = 0
        cuda_state = torch.cuda.get_rng_state(device_index)
    else:
        cuda_state = None
    if distributed_context["world_size"] > 1:
        cuda_states = all_gather_list(cuda_state)
    else:
        cuda_states = [cuda_state]
    rng_state = {
        "cpu_state": cpu_state,
        "cuda_states": cuda_states,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    return rng_state


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
        trainer_state: Optional[dict] = None,
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
        self.trainer_state = trainer_state or {}
        if "data_state" in self.trainer_state:
            logger.info("Restoring data gateway state")
            self.train_dataloader.set_state(self.trainer_state["data_state"])
        else:
            self.trainer_state.update(
                best_checkpoint=None,
                best_metric=float("-inf") * int(self.config.greater_is_better),
            )

    def checkpoint_state(self):
        return copy.deepcopy(self.trainer_state)

    def fetch_batch(self, func_tracker: Optional[dict] = None):
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
            chunk_tracker = {}
            chunked_unique_ids_and_tokens_ids = equal_chunking(
                unique_ids_and_tokens_ids,
                n_chunks=distributed_context["world_size"],
                tracker=chunk_tracker,
            )[distributed_context["local_rank"]]
            if isinstance(tracker, dict):
                func_tracker["added"] = chunk_tracker["added"][
                    distributed_context["local_rank"]
                ]
        else:
            chunked_unique_ids_and_tokens_ids = unique_ids_and_tokens_ids
            if isinstance(tracker, dict):
                func_tracker["added"] = 1

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
        if (
            distributed_context["local_rank"] == -1
            or distributed_context["local_rank"] == 0
        ):
            for reporter in self.reporters:
                reporter.init_run(
                    project_name=self.config.wandb_project,
                    config=self.config.model_dump(),
                )

        trained_steps = self.trainer_state.get("trained_steps", 0)
        if trained_steps > 0:
            logger.info("Model was trained for {} steps".format(trained_steps))

        with self.train_dataloader.pipeline_context():
            t0 = time.perf_counter()
            for step in range(trained_steps, self.config.num_train_steps):
                tracker = {}
                batch = self.fetch_batch(func_tracker=tracker)
                loss = self.train_step(
                    batch, added=tracker["added"]
                )  # scalar loss value

                # state update and log at the end of the iteration
                trained_steps = step + 1
                self.trainer_state["trained_steps"] = trained_steps
                if trained_steps % self.config.logging_steps == 0:
                    logger.info(
                        "Step {}/{}: Loss = {} - Elapsed time: {}s".format(
                            trained_steps,
                            self.config.num_train_steps,
                            loss,
                            time.perf_counter() - t0,
                        )
                    )
                    t0 = time.perf_counter()
                    if (
                        distributed_context["local_rank"] == -1
                        or distributed_context["local_rank"] == 0
                    ):
                        for reporter in self.reporters:
                            reporter.log({"train/loss": loss}, step=trained_steps)

                if self.config.do_eval and trained_steps % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    metric = metrics[self.config.metric_for_best_model]
                    if self.trainer_state["best_metric"] * int(
                        self.config.greater_is_better
                    ) < metric * int(self.config.greater_is_better):
                        self.trainer_state["best_metric"] = metric
                        self.trainer_state["best_checkpoint"] = (
                            "checkpoint-{:06d}".format(
                                self.trainer_state["trained_steps"]
                            )
                        )
                if trained_steps % self.config.save_steps == 0:
                    rng_state = save_rng_state()
                    if (
                        distributed_context["local_rank"] == -1
                        or distributed_context["local_rank"] == 0
                    ):
                        self.save_checkpoint(rng_state)

    def train_step(self, batch, added: int = 1):
        logger.debug("Local input tensor size: {}".format(batch["input_ids"].size()))
        self.optimizer.zero_grad()

        if self.config.grad_cache.use is True:
            list_inputs = []
            for input_ids, attn_mask in zip(
                batch["input_ids"], batch["attention_mask"]
            ):
                list_inputs.append((input_ids, attn_mask))
            list_chunks = equal_chunking_with_maxsize(
                list_inputs, self.config.grad_cache.max_chunk_size
            )
            _list_chunks = []
            for chunk in list_chunks:
                chunked_input_ids = []
                chunked_attn_mask = []
                for input_ids, attn_mask in chunk:
                    chunked_input_ids.append(input_ids)
                    chunked_attn_mask.append(attn_mask)
                chunked_input_ids = torch.stack(chunked_input_ids, dim=0)
                chunked_attn_mask = torch.stack(chunked_attn_mask, dim=0)
                _list_chunks.append((chunked_input_ids, chunked_attn_mask))
            list_chunks = _list_chunks

            # forward no grad
            rand_contexts = []
            list_chunked_scores = []
            with torch.no_grad():
                for chunked_input_ids, chunked_attn_mask in list_chunks:
                    rand_contexts.append(
                        RandContext(chunked_input_ids, chunked_attn_mask)
                    )
                    chunked_scores = self.model(
                        input_ids=chunked_input_ids, attention_mask=chunked_attn_mask
                    )
                    list_chunked_scores.append(chunked_scores)
            for chunked_scores in list_chunked_scores:
                chunked_scores.requires_grad_()

            local_score_map = {}
            for unique_id, score in zip(
                batch["unique_ids"], itertools.chain(*list_chunked_scores)
            ):
                local_score_map[unique_id] = score

            global_score_map = self.gather_score_map(local_score_map)
            step_loss = self.calculate_loss(
                global_score_map, batch["contrastive_samples"]
            )
            scalar_step_loss = step_loss.item()
            _step_loss = (
                step_loss * distributed_context["world_size"] * self.config.score_scale
            )
            _step_loss.backward()

            list_chunked_scores_grads = [scores.grad for scores in list_chunked_scores]
            for (input_ids, attn_mask), grad, rnd_ctx in zip(
                list_chunks, list_chunked_scores_grads, rand_contexts
            ):
                with rnd_ctx:
                    chunked_scores = self.model(
                        input_ids=input_ids, attention_mask=attn_mask
                    )
                    surrogate = torch.dot(chunked_scores.flatten(), grad.flatten())
                    surrogate.backward()
            if added == 0:
                n_chunks = len(list_chunks)
                possible_n_chunks = (
                    len(list_inputs) // self.config.grad_cache.max_chunk_size + 1
                )
                if n_chunks < possible_n_chunks:
                    # dummy forward backward
                    dummy_input_ids = torch.tensor(
                        [[0]], dtype=torch.int64, device=distributed_context["device"]
                    )
                    dummy_attn_mask = torch.tensor(
                        [[0]], dtype=torch.int64, device=distributed_context["device"]
                    )
                    dummy_score = self.model(
                        input_ids=dummy_input_ids, attention_mask=dummy_attn_mask
                    )
                    zero_grad = torch.zeros_like(dummy_score)
                    surrogate = torch.dot(dummy_score.flatten(), zero_grad.flatten())
                    surrogate.backward()
        else:
            scores = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )  # [bsz]
            local_score_map = {}
            for i, unique_id in enumerate(batch["unique_ids"]):
                local_score_map[unique_id] = scores[i]

            global_score_map = self.gather_score_map(local_score_map)
            step_loss = self.calculate_loss(
                global_score_map, batch["contrastive_samples"]
            )

            scalar_step_loss = step_loss.item()
            _step_loss = (
                step_loss * distributed_context["world_size"] * self.config.score_scale
            )
            _step_loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        return scalar_step_loss

    def gather_score_map(self, local_score_map: dict[str, torch.Tensor]):
        if distributed_context["world_size"] > 1:
            all_local_score_maps = all_gather_list(
                local_score_map, max_size=self.config.gather_buf_size
            )
            global_score_map = {}
            for score_map in all_local_score_maps:
                for k, v in score_map.items():
                    if k in local_score_map:
                        global_score_map[k] = local_score_map[k]
                    else:
                        global_score_map[k] = v.detach().to(
                            distributed_context["device"]
                        )
        else:
            global_score_map = local_score_map
        return global_score_map

    def calculate_loss(
        self, score_map: dict[str, torch.Tensor], contrastive_samples: dict[str, list]
    ):
        loss = []
        for positive_id, negatives_ids in contrastive_samples.items():
            pos_score = score_map[positive_id]
            negs_scores = [score_map[neg_id] for neg_id in negatives_ids]
            pos_and_negs_scores = torch.stack([pos_score] + negs_scores, dim=0)
            logits = F.log_softmax(pos_and_negs_scores, dim=-1)
            loss.append(logits[0] * -1)
        loss = torch.stack(loss)
        loss = torch.mean(loss)
        return loss

    def transform_eval_batch(self, batch):
        # chunk batch
        lookup = batch["lookup"]
        unique_ids_and_tokens_ids = []
        for k, v in lookup.items():
            unique_ids_and_tokens_ids.append((k, v["concat_tokens_ids"]))
        if distributed_context["world_size"] > 1:
            chunk_tracker = {}
            chunked_unique_ids_and_tokens_ids = equal_chunking(
                unique_ids_and_tokens_ids,
                n_chunks=distributed_context["world_size"],
                tracker=chunk_tracker,
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
        }

    def evaluate(self):
        logger.info("Running evaluation")
        self.model.eval()

        total = Counter()
        matches = Counter()
        progress_bar = tqdm(desc="Batch")
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in self.eval_dataloader:
                items = batch["items"]
                batch = self.transform_eval_batch(batch)
                if self.config.grad_cache.use is True:
                    list_inputs = []
                    for input_ids, attn_mask in zip(
                        batch["input_ids"], batch["attention_mask"]
                    ):
                        list_inputs.append((input_ids, attn_mask))
                    list_chunks = equal_chunking_with_maxsize(
                        list_inputs, self.config.grad_cache.max_chunk_size
                    )
                    _list_chunks = []
                    for chunk in list_chunks:
                        chunked_input_ids = []
                        chunked_attn_mask = []
                        for input_ids, attn_mask in chunk:
                            chunked_input_ids.append(input_ids)
                            chunked_attn_mask.append(attn_mask)
                        chunked_input_ids = torch.stack(chunked_input_ids, dim=0)
                        chunked_attn_mask = torch.stack(chunked_attn_mask, dim=0)
                        _list_chunks.append((chunked_input_ids, chunked_attn_mask))
                    list_chunks = _list_chunks

                    # forward no grad
                    list_chunked_scores = []
                    for chunked_input_ids, chunked_attn_mask in list_chunks:
                        chunked_scores = self.model(
                            input_ids=chunked_input_ids,
                            attention_mask=chunked_attn_mask,
                        )
                        list_chunked_scores.append(chunked_scores)

                    local_score_map = {}
                    for unique_id, score in zip(
                        batch["unique_ids"], itertools.chain(*list_chunked_scores)
                    ):
                        local_score_map[unique_id] = score.item()
                else:
                    scores = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )  # [bsz]
                    local_score_map = {}
                    for i, unique_id in enumerate(batch["unique_ids"]):
                        local_score_map[unique_id] = scores[i].item()

                if distributed_context["world_size"] > 1:
                    all_local_score_maps = all_gather_list(
                        local_score_map, max_size=self.config.gather_buf_size
                    )
                    global_score_map = {}
                    for score_map in all_local_score_maps:
                        for k, v in score_map.items():
                            global_score_map[k] = v
                else:
                    global_score_map = local_score_map

                for item in items:
                    total["all"] += 1
                    total[item["type"]] += 1
                    if (
                        global_score_map[item["positive"]["unique_id"]]
                        > global_score_map[item["negative"]["unique_id"]]
                    ):
                        matches["all"] += 1
                        matches[item["type"]] += 1

                progress_bar.update(1)

        metrics = {}
        for pipeline_type in matches:
            acc = matches[pipeline_type] / total[pipeline_type]
            if pipeline_type == "all":
                metrics["eval/acc"] = acc
            else:
                metrics["eval/acc/{}".format(pipeline_type)] = acc
            if (
                distributed_context["local_rank"] == -1
                or distributed_context["local_rank"] == 0
            ):
                for reporter in self.reporters:
                    reporter.log(metrics, step=self.trainer_state["trained_steps"])

        self.model.train()
        logger.info("Evaluation done in {}s".format(time.perf_counter() - t0))
        logger.info("Metrics: {}".format(metrics))
        return metrics

    def save_checkpoint(self, rng_state):
        logger.info("Saving checkpoint...")
        checkpoint_dir = os.path.join(self.config.output_dir, self.run_id)

        model_to_save = get_model_obj(self.model)
        cp_name = "checkpoint-{:06d}".format(self.trainer_state["trained_steps"])
        cp_path = os.path.join(checkpoint_dir, cp_name)

        all_cp_paths = os.listdir(checkpoint_dir)
        all_cp_paths = [f for f in all_cp_paths if f.startswith("checkpoint-")]
        all_cp_paths = [os.path.join(checkpoint_dir, f) for f in all_cp_paths]
        all_cp_paths = sorted(
            all_cp_paths, key=lambda x: os.stat(x).st_mtime, reverse=True
        )
        best_cp_path = []
        other_cp_paths = []
        for cp in all_cp_paths:
            if "best_checkpoint" in self.trainer_state and cp == os.path.join(
                checkpoint_dir, self.trainer_state["best_checkpoint"]
            ):
                best_cp_path.append(cp)
            else:
                other_cp_paths.append(cp)
        if len(best_cp_path) == 1 and best_cp_path[0] == cp_path:
            head = [cp_path]
        else:
            head = best_cp_path + [cp_path]
        all_cp_paths = head + other_cp_paths

        if self.config.save_total_limit > 0:
            paths_to_delete = all_cp_paths[self.config.save_total_limit :]
            for path in paths_to_delete:
                logger.info("Deleting {}".format(path))
                shutil.rmtree(path)

        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        with open(os.path.join(cp_path, "model.pth"), "wb") as writer:
            torch.save(model_to_save.state_dict(), writer)
        with open(os.path.join(cp_path, "optimizer.pt"), "wb") as writer:
            torch.save(self.optimizer.state_dict(), writer)
        with open(os.path.join(cp_path, "scheduler.pt"), "wb") as writer:
            torch.save(self.scheduler.state_dict(), writer)
        with open(
            os.path.join(cp_path, "trainer_state.json"), "w", encoding="utf-8"
        ) as writer:
            json.dump(self.trainer_state, writer, indent=4, ensure_ascii=False)
        with open(os.path.join(cp_path, "rng_state.pth"), "wb") as writer:
            torch.save(rng_state, writer)

        logger.info("Checkpoint saved to {}".format(cp_path))
