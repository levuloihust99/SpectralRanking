import json
import os
import yaml
import random
import string
import logging
import argparse
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, T5ForConditionalGeneration

from lib.nn.modeling import T5CrossEncoder
from lib.nn.optimization import get_optimizer, get_schedule_linear
from lib.utils.logging import do_setup_logging
from lib.utils.seeding import seed_everything

from lib.v2.nn.ctx import distributed_context
from lib.v2.nn.configuration import CrossEncoderConfig
from lib.v2.data_helpers.gateway import DataGateway
from lib.v2.data_helpers.eval_dataloader import EvalDataLoader
from lib.v2.nn.trainer import CrossEncoderTrainer

do_setup_logging()
logger = logging.getLogger(__name__)


def find_device():
    if torch.cuda.is_available():
        if distributed_context["local_rank"] != -1:
            device = torch.device("cuda:{}".format(distributed_context["local_rank"]))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    distributed_context["device"] = device


@contextmanager
def may_setup_distributed():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        distributed_context["local_rank"] = -1
        distributed_context["world_size"] = 1
        yield
    else:
        local_rank = int(local_rank)
        world_size = int(os.environ.get("WORLD_SIZE"))
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        distributed_context["local_rank"] = local_rank
        distributed_context["world_size"] = world_size
        yield
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=".config.yml")
    args = parser.parse_args()

    # READ CONFIG
    with open(args.config_file, "r", encoding="utf-8") as reader:
        config_dict = yaml.safe_load(reader)
    config = CrossEncoderConfig(**config_dict)

    # < SETUP LOGGING AND RANDOM
    run_id = "".join(
        random.choice(string.digits + string.ascii_uppercase) for _ in range(16)
    )
    do_setup_logging(level=config.log_level)
    if config.seed:
        seed_everything(config.seed)
    # SETUP LOGGING AND RANDOM />

    # < WRITE CONFIG
    output_dir = os.path.join(config.output_dir, run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(
        os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8"
    ) as writer:
        json.dump(config.model_dump(), writer, indent=4, ensure_ascii=False)

    # < INIT TOKENIZER AND MODEL
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    pretrained = T5ForConditionalGeneration.from_pretrained(config.model_path)
    model = T5CrossEncoder.from_t5_for_conditional_generation(pretrained)
    # INIT TOKENIZER AND MODEL />

    # SETUP MODEL
    model.to(distributed_context["device"])
    optimizer = get_optimizer(
        model,
        learning_rate=config.learning_rate,
        adam_eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    if distributed_context["local_rank"] != -1:
        model = DDP(
            model,
            device_ids=[distributed_context["local_rank"]],
            output_device=distributed_context["local_rank"],
        )

    # < SETUP DATA PIPELINE
    train_dataloader = DataGateway(config.data_config)
    if config.do_eval:
        eval_dataloader = EvalDataLoader(config.eval_data_config)
    # SETUP DATA PIPELINE />

    # < SCHEDULER
    total_steps = config.num_train_steps
    num_warmup_steps_by_ratio = int(total_steps * config.warmup_ratio)
    num_warmup_steps_absolute = config.warmup_steps
    if num_warmup_steps_absolute == 0 or num_warmup_steps_by_ratio == 0:
        num_warmup_steps = max(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    else:
        num_warmup_steps = min(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    scheduler = get_schedule_linear(
        optimizer, warmup_steps=num_warmup_steps, training_steps=total_steps
    )
    # SCHEDULER />

    # < LOAD CHECKPOINT
    if config.resume_from_checkpoint:
        pass
    # LOAD CHECKPOINT />

    # < TRAINER
    trainer = CrossEncoderTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        run_id=run_id,
    )
    trainer.train()
    # TRAINER />


if __name__ == "__main__":
    with may_setup_distributed():
        main()
