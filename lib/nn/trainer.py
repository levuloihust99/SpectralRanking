import os
import torch
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)

from tqdm import tqdm

from lib.nn.configuration import CrossEncoderConfig
from lib.nn.losses import SpectralRankingLoss
from lib.utils.reporters import WandBReporter


class CrossEncoderTrainer:
    def __init__(
        self,
        config: CrossEncoderConfig,
        model: torch.nn.Module,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        training_state,
        rng_states,
        run_id
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(config.gpu_id))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(config.gpu_id), torch.cuda.get_device_capability(config.gpu_id)))
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     logger.info("MPS backend is available, using MPS.")
        else:
            logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_state = training_state
        self.rng_states = rng_states
        self.loss_calculator = SpectralRankingLoss(device=self.device, margin=config.loss_margin * config.score_scale)
        self.run_id = run_id

        self.reporters = []
        for reporter_name in config.report_to:
            if reporter_name == "wandb":
                self.reporters.append(WandBReporter(api_key=config.wandb_api_key, run_id=run_id))

    def train(self):
        self.model.train()
        logger.info("*************** Start training ***************")
        for reporter in self.reporters:
            reporter.init_run(project_name="SpectralRanking", config=self.config.to_json())

        trained_steps = 0
        trained_epoch = 0
        if self.training_state:
            trained_steps = self.training_state.get('step', 0)
            trained_epoch = self.training_state.get('epoch', 0)

        if trained_steps == len(self.train_dataloader):
            trained_epoch += 1
            trained_steps = 0

        best_checkpoint = None
        best_metric = float("-inf") * int(self.config.greater_is_better)
        if self.training_state:
            best_checkpoint = self.training_state.get("best_checkpoint")
            best_metric = self.training_state.get("best_metric")

        for epoch in range(trained_epoch, self.config.num_train_epochs):
            logger.info(' Epoch {:} / {:}'.format(epoch + 1, self.config.num_train_epochs))
            self.train_dataloader.sampler.set_epoch(trained_epoch)
            iterator = iter(self.train_dataloader)
            for i in range(trained_steps):
                next(iterator)
            if trained_steps > 0: # this is a resume
                torch.random.set_rng_state(self.rng_states["cpu_state"])
                if torch.cuda.is_available():
                    torch.cuda.random.set_rng_state(self.rng_states["cuda_state"])
            total = len(self.train_dataloader)
            count = 0
            for batch in tqdm(iterator, total=total, leave=trained_steps, initial=trained_steps):
                step = count + trained_steps
                count += 1
                normalization = sum(len(item["comparisons"]) for item in batch)
                batch_loss = 0.0
                for sample in batch:
                    generators = sample["generators"]
                    inputs = {k: torch.tensor(v, device=self.device) for k, v in sample["inputs"].items()}
                    scores = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]) # [bsz]
                    score_map = {}
                    for i in range(len(generators)):
                        score_map[generators[i]] = scores[i] * self.config.score_scale
                    loss = self.loss_calculator.calculate(score_map=score_map, comparisons=sample["comparisons"])
                    loss /= normalization
                    batch_loss += loss.item()
                    loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                if (step + 1) % self.config.logging_steps == 0:
                    for reporter in self.reporters:
                        reporter.log({"train/loss": batch_loss}, step=step + 1)

                if (step + 1) % self.config.eval_steps == 0:
                    metric = self.evaluate(step + 1)
                    if (step + 1) % self.config.save_steps == 0:
                        cp_name = 'checkpoint-{:06d}.pth'.format(step + 1)
                        if best_metric * int(self.config.greater_is_better) < metric * int(self.config.greater_is_better):
                            best_metric = metric
                            best_checkpoint = cp_name

                        training_state = {
                            "epoch": epoch,
                            "step": step + 1,
                            "best_checkpoint": best_checkpoint,
                            "best_metric": best_metric
                        }
                        self.save_checkpoint(cp_name, training_state)

            trained_steps = 0

    def evaluate(self, step):
        logger.info("Running evaluation...")
        self.model.eval()
        total = 0
        matches = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                for sample in batch:
                    total += len(sample["comparisons"])
                    inputs = {k: torch.tensor(v, device=self.device) for k, v in sample["inputs"].items()}
                    scores = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    generators = sample["generators"]
                    score_map = {}
                    for i in range(len(generators)):
                        score_map[generators[i]] = scores[i].item()
                    for pair, preferred in sample["comparisons"].items():
                        g1, g2 = eval(pair)
                        is_match = (score_map[g1] - score_map[g2]) * preferred
                        if is_match > 0:
                            matches += 1
        acc = matches / total
        for reporter in self.reporters:
            reporter.log({"eval/acc": acc}, step=step)
        self.model.train()
        return acc

    def save_checkpoint(self, cp_name, training_state):
        checkpoint_dir = os.path.join(self.config.output_dir, self.run_id)

        # < save RNG state
        cpu_state = torch.get_rng_state()
        if self.device.type == 'cuda':
            cuda_state = torch.cuda.get_rng_state()
        else:
            cuda_state = None
        rng_states = {'cpu_state': cpu_state, 'cuda_state': cuda_state, 'python': random.getstate(), 'numpy': np.random.get_state()}
        # save RNG state />

        cp_path = os.path.join(checkpoint_dir, cp_name)

        saved_state = {
            'model_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'run_id': self.run_id,
            'rng_states': rng_states,
            'training_state': training_state
        }

        all_cp_files = os.listdir(checkpoint_dir)
        all_cp_files = [f for f in all_cp_files if f.startswith('checkpoint-')]
        all_cp_files = [os.path.join(checkpoint_dir, f) for f in all_cp_files]
        all_cp_files = sorted(all_cp_files, key=lambda x: os.stat(x).st_mtime, reverse=True)
        best_checkpoint_file = []
        other_checkpoint_files = []
        for cp in all_cp_files:
            if cp == os.path.join(checkpoint_dir, training_state["best_checkpoint"]):
                best_checkpoint_file.append(cp)
            else:
                other_checkpoint_files.append(cp)
        if len(best_checkpoint_file) == 1 and best_checkpoint_file[0] == cp_path:
            head = [cp_path]
        else:
            head = best_checkpoint_file + [cp_path]
        all_cp_files = head + other_checkpoint_files

        if self.config.save_total_limit > 0:
            files_to_delete = all_cp_files[self.config.save_total_limit:]
            for f in files_to_delete:
                os.remove(f)
        with open(cp_path, "wb") as writer:
            torch.save(saved_state, writer)
        logger.info("Saved checkpoint at {}".format(cp_path))
