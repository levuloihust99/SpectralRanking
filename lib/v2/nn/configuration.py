from typing import Literal, Optional, Any
from pydantic import BaseModel, model_validator, PositiveInt

from ..data_helpers.schemas import EvalDataConfig
from ..data_helpers.gateway import DataGatewayConfig


class GradCacheConfig(BaseModel):
    use: bool = False
    max_chunk_size: PositiveInt = 2


class CrossEncoderConfig(BaseModel):
    # tokenizer
    tokenizer_path: str = "VietAI/vit5-base"
    sep_token: str = "<extra_id_0>"

    # model
    type_model: str = "t5"
    path_to_model: Optional[str] = None

    # data pipeline
    data_config: DataGatewayConfig
    eval_data_config: Optional[EvalDataConfig] = None
    
    grad_cache: GradCacheConfig

    # training
    output_dir: str = "assets/outputs"
    do_train: bool = True
    do_eval: bool = True
    score_scale: float = 1.0
    learning_rate: float = 2e-5
    num_train_steps: int = 20000
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    metric_for_best_model: str = "eval/acc"
    greater_is_better: bool = True
    resume_from_checkpoint: Optional[str] = None
    seed: Optional[int] = 12345

    # data
    max_input_len: int = 1500

    # log config
    log_level: Literal["info", "warning", "debug", "error"] = "info"
    report_to: list[str] = ["wandb"]
    wandb_api_key: Optional[str] = None
    wandb_project: Optional[str] = "SpectralRankingV2"
    logging_dir: str = "assets/logs"
    logging_steps: int = 10
    ggdrive_storage_folder_id: Optional[str] = None
    
    gather_buf_size: Optional[PositiveInt] = 16384

    @model_validator(mode="before")
    @classmethod
    def inject_seed_to_data_config(cls, data: Any):
        data["data_config"]["seed"] = data["seed"]
        return data

    @model_validator(mode="after")
    def validate_config(self):
        if self.do_eval and self.save_steps % self.eval_steps != 0:
            raise ValueError("`save_steps` must be multiple of `eval_steps`")
        return self
