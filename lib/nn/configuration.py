import logging

logger = logging.getLogger(__name__)


class CrossEncoderConfig:
    def __init__(self, **kwargs):
        # tokenizer
        self.tokenizer_path = "VietAI/vit5-base"
        self.sep_token = '<extra_id_0>'

        # model
        self.model_type = "t5"
        self.model_path = None

        # dataset
        self.train_data_path = None
        self.eval_data_path = None

        # training config
        self.gpu_id = 0
        self.output_dir = "assets/outputs"
        self.do_train = True
        self.do_eval = True
        self.score_scale = 5.0
        self.loss_margin = 0.05
        self.learning_rate = 2e-5
        self.num_train_epochs = 10
        self.warmup_ratio = 0.0
        self.warmup_steps = 0
        self.weight_decay = 0.0
        self.per_device_train_batch_size = 2
        self.per_device_eval_batch_size = 2
        self.adam_eps = 1e-8
        self.save_steps = 100
        self.eval_steps = 100
        self.save_total_limit = 2
        self.max_grad_norm = 1.0
        self.metric_for_best_model = "eval/acc"
        self.greater_is_better = True
        self.resume_from_checkpoint = None
        self.seed = 12345

        # data config
        self.max_input_len = 1500

        # log config
        self.log_level = "info"
        self.report_to = ["wandb"]
        self.wandb_api_key = None
        self.logging_dir = "assets/logs"
        self.logging_steps = 1

        self.override_defaults(**kwargs)
        self.validate_config()

    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v

    def validate_config(self):
        assert self.save_steps % self.eval_steps == 0
    
    def to_json(self):
        return {
            # tokenizer
            "tokenizer_path": self.tokenizer_path,
            "sep_token": self.sep_token,
            # model
            "model_type": self.model_type,
            "model_path": self.model_path,
            # dataset
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            # training config
            "gpu_id": self.gpu_id,
            "output_dir": self.output_dir,
            "do_train": self.do_train,
            "do_eval": self.do_eval,
            "score_scale": self.score_scale,
            "loss_margin": self.loss_margin,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "adam_eps": self.adam_eps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "max_grad_norm": self.max_grad_norm,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "seed": self.seed,
            # data config,
            "max_input_len": self.max_input_len,
            # log config
            "log_level": self.log_level,
            "report_to": self.report_to,
            "wandb_api_key": self.wandb_api_key,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
        }
