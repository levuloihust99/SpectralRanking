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

from transformers import T5ForConditionalGeneration

from lib.nn.modeling import T5CrossEncoder
from lib.nn.optimization import get_optimizer, get_schedule_linear
from lib.utils.logging import do_setup_logging
from lib.utils.seeding import seed_everything

from lib.v2.nn.ctx import distributed_context
from lib.v2.nn.configuration import CrossEncoderConfig
from lib.v2.data_helpers.gateway import DataGateway
from lib.v2.data_helpers.eval_dataloader import EvalDataLoader
from lib.v2.nn.trainer import CrossEncoderTrainer, get_model_obj

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
        find_device()
        yield
    else:
        local_rank = int(local_rank)
        world_size = int(os.environ.get("WORLD_SIZE"))
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        distributed_context["local_rank"] = local_rank
        distributed_context["world_size"] = world_size
        find_device()
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
    pretrained = T5ForConditionalGeneration.from_pretrained(config.path_to_model)
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
        eval_dataloader = EvalDataLoader(config.eval_data_config, config.data_config)
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
    rng_state = None
    trainer_state = None
    if config.resume_from_checkpoint:
        cp_path = config.resume_from_checkpoint
        logger.info("Loading checkpoint from {}".format(cp_path))

        run_id = os.path.basename(os.path.dirname(cp_path))
        with open(os.path.join(cp_path, "trainer_state.json"), "r") as reader:
            trainer_state = json.load(reader)

        rng_state = torch.load(os.path.join(cp_path, "rng_state.pth"))
        # restore RNG
        cpu_state = rng_state.get("cpu_state")
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        cuda_states = rng_state.get("cuda_states")
        if cuda_states is not None:
            if torch.cuda.is_available() > 0:
                local_rank = distributed_context["local_rank"]
                if local_rank < len(cuda_states):
                    try:
                        torch.cuda.set_rng_state(
                            cuda_states[local_rank],
                            device=local_rank,
                        )
                    except Exception:
                        logger.error(
                            "Invalid RNG state restored from checkpoint file '{}'".format(
                                cp_path
                            )
                        )

        model_state = torch.load(
            os.path.join(cp_path, "model.pth"), map_location=lambda s, t: s
        )
        get_model_obj(model).load_state_dict(model_state)
        optimizer_state = torch.load(
            os.path.join(cp_path, "optimizer.pt"), map_location=lambda s, t: s
        )
        optimizer.load_state_dict(optimizer_state)
        scheduler_state = torch.load(os.path.join(cp_path, "scheduler.pt"))
        scheduler.load_state_dict(scheduler_state)
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
        trainer_state=trainer_state,
    )
    trainer.train()
    # TRAINER />


if __name__ == "__main__":
    with may_setup_distributed():
        main()
