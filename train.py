import os
import json
import copy
import torch
import random
import string
import logging
import numpy as np

logger = logging.getLogger(__name__)

from typing import Text

from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from lib.utils.arguments import create_parser
from lib.utils.seeding import seed_everything
from lib.utils.logging import do_setup_logging
from lib.data_helpers.bytedataset import ByteDataset
from lib.data_helpers.dataloader import get_collate_fn
from lib.nn.modeling import T5CrossEncoder
from lib.nn.configuration import CrossEncoderConfig
from lib.nn.optimization import get_optimizer, get_schedule_linear
from lib.nn.trainer import CrossEncoderTrainer


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


def config_logging(level: Text):
    log_level = None
    if level.lower() == "info":
        log_level = logging.INFO
    elif level.lower() == "debug":
        log_level = logging.DEBUG
    elif level.lower() == "warning":
        log_level = logging.WARNING
    elif level.lower() == "error":
        log_level = logging.ERROR
    elif level.lower() == "critical":
        log_level = logging.CRITICAL
    else:
        log_level = logging.NOTSET
    logging.basicConfig(level=log_level)


def main():
    parser = create_parser()
    args = parser.parse_args()
    args_json = copy.deepcopy(args.__dict__)
    hparams = args_json.pop('hparams')
    if args.hparams.endswith('.json'):
        with open(args.hparams, "r") as f:
            hparams = json.load(f)
    else:
        hparams = json.loads(args.hparams)
    hparams = override_defaults(hparams, args_json)
    cfg = CrossEncoderConfig(**hparams)

    run_id = ''.join(random.choice(string.digits + string.ascii_uppercase) for _ in range(16))
    config_logging(level=cfg.log_level)
    do_setup_logging()
    if cfg.seed:
        seed_everything(cfg.seed)

    output_dir = os.path.join(cfg.output_dir, run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "training_config.json"), "w") as writer:
        json.dump(cfg.to_json(), writer, indent=4, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    pretrained = T5ForConditionalGeneration.from_pretrained(cfg.model_path)
    model = T5CrossEncoder.from_t5_for_conditional_generation(pretrained)
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cfg.gpu_id))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(cfg.gpu_id), torch.cuda.get_device_capability(cfg.gpu_id)))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS backend is available, using MPS.")
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    model.to(device)
    optimizer = get_optimizer(
        model, learning_rate=cfg.learning_rate, adam_eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

    # dataloader
    train_dataset = ByteDataset(cfg.train_data_path, 6)
    sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True, seed=cfg.seed)
    train_data_collate_fn = get_collate_fn(
        tokenizer=tokenizer,
        sep_token=cfg.sep_token,
        max_input_len=cfg.max_input_len
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=train_data_collate_fn
    )
    if cfg.do_eval:
        eval_dataset = ByteDataset(cfg.eval_data_path, 6)
        eval_data_collate_fn = get_collate_fn(
            tokenizer=tokenizer,
            sep_token=cfg.sep_token
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=cfg.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=eval_data_collate_fn
        )
    else:
        eval_dataloader = None

    # scheduler
    total_steps = len(train_dataloader) * cfg.num_train_epochs
    num_warmup_steps_by_ratio = int(total_steps * cfg.warmup_ratio)
    num_warmup_steps_absolute = cfg.warmup_steps
    if num_warmup_steps_absolute == 0 or num_warmup_steps_by_ratio == 0:
        num_warmup_steps = max(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    else:
        num_warmup_steps = min(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    scheduler = get_schedule_linear(optimizer, warmup_steps=num_warmup_steps, training_steps=total_steps)

    # restore checkpoint state
    training_state = None
    rng_states = None
    if cfg.resume_from_checkpoint:
        saved_state = torch.load(cfg.resume_from_checkpoint)
        logger.info("Loading saved model state...")
        model.load_state_dict(saved_state["model_dict"])
        logger.info("Loading saved optimizer state...")
        optimizer.load_state_dict(saved_state["optimizer_dict"])
        logger.info("Loading saved scheduler state...")
        scheduler.load_state_dict(saved_state["scheduler_dict"])
        training_state = saved_state["training_state"]
        run_id = saved_state["run_id"]

        # < restore RNG
        rng_states = saved_state.get('rng_states', {})
        python_rng_state = rng_states.get('python', None)
        if python_rng_state:
            random.setstate(python_rng_state)
        numpy_rng_state = rng_states.get('numpy', None)
        if numpy_rng_state:
            np.random.set_state(numpy_rng_state)

        cpu_state = rng_states.get('cpu_state')
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        
        cuda_state = rng_states.get('cuda_state', None)
        if cuda_state is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_state)
        # restore RNG />

    # trainer
    training_info = {
        "warmup_steps": num_warmup_steps,
        "total_steps": total_steps
    }
    trainer = CrossEncoderTrainer(
        config=cfg,
        device=device,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        rng_states=rng_states,
        run_id=run_id,
        training_info=training_info
    )
    trainer.train()


if __name__ == "__main__":
    main()
