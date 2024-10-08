import copy
from enum import Enum
from typing import Any, Literal, Optional, Union, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator, PositiveInt


class PipelineType(str, Enum):
    SLIDING = "sliding"
    NON_CONCISENESS = "nonconciseness"
    NON_COHERENCE = "noncoherence"
    NON_COVERAGE = "noncoverage"
    MODEL_COMPARE = "model_compare"
    NEGPOOL = "negpool"
    NLI_SLIDING = "nli_sliding"
    NLI_STRATIFIED = "nli_stratified"

    def __str__(self):
        return self.value


class SlidingConfig(BaseModel):
    type: Literal[PipelineType.SLIDING] = PipelineType.SLIDING

    dataset_path: str
    buffer_size: PositiveInt
    contrastive_size: PositiveInt


class NonconcisenessConfig(BaseModel):
    type: Literal[PipelineType.NON_CONCISENESS] = PipelineType.NON_CONCISENESS

    dataset_path: str
    buffer_size: PositiveInt


class NoncoherenceConfig(BaseModel):
    type: Literal[PipelineType.NON_COHERENCE] = PipelineType.NON_COHERENCE

    dataset_path: str
    buffer_size: PositiveInt


class NoncoverageConfig(BaseModel):
    type: Literal[PipelineType.NON_COVERAGE] = PipelineType.NON_COVERAGE

    dataset_path: str
    buffer_size: PositiveInt


class ModelCompareConfig(BaseModel):
    type: Literal[PipelineType.MODEL_COMPARE] = PipelineType.MODEL_COMPARE

    dataset_path: str
    buffer_size: PositiveInt


class NegPoolConfig(BaseModel):
    type: Literal[PipelineType.NEGPOOL] = PipelineType.NEGPOOL

    pos_dataset_path: str
    neg_dataset_path: str
    buffer_size: PositiveInt
    contrastive_size: PositiveInt


class NLISlidingConfig(BaseModel):
    type: Literal[PipelineType.NLI_SLIDING] = PipelineType.NLI_SLIDING

    dataset_path: str
    buffer_size: PositiveInt
    contrastive_size: PositiveInt


class NLIStratifiedConfig(BaseModel):
    type: Literal[PipelineType.NLI_STRATIFIED] = PipelineType.NLI_STRATIFIED

    dataset_path: str
    buffer_size: PositiveInt


PipelineConfig = Annotated[
    Union[
        SlidingConfig,
        NonconcisenessConfig,
        NoncoherenceConfig,
        NoncoverageConfig,
        ModelCompareConfig,
        NegPoolConfig,
        NLISlidingConfig,
        NLIStratifiedConfig,
    ],
    Field(discriminator="type"),
]


class DataGatewayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    main_data_path: Optional[str] = Field(
        default=None,
        description="Path to the bytedataset for all pipelines and positive pipeline of NegPoolPipeline.",
    )
    negpool_data_path: Optional[str] = Field(
        default=None,
        description="Path to the negative bytedataset for NegPoolPipeline.",
    )
    max_input_len: Optional[int] = Field(
        default=None,
        description="Maximum input length in tokens (max tokens for document).",
    )
    max_output_len: Optional[int] = Field(
        default=None,
        description="Maximum output length in tokens (max tokens for summaries).",
    )
    regulate_factors: dict[str, PositiveInt] = Field(
        ...,
        description=(
            "Specify how many consecutive times that the gateway take from a specific pipeline."
        ),
        examples={
            PipelineType.SLIDING: 2,
            PipelineType.NON_CONCISENESS: 3,
            PipelineType.NON_COHERENCE: 1,
            PipelineType.NON_COVERAGE: 1,
            PipelineType.MODEL_COMPARE: 1,
            PipelineType.NEGPOOL: 1,
        },
    )
    buffer_size: Optional[PositiveInt] = Field(
        default=None,
        description=(
            "Common buffer size used for all pipelines. "
            "If a pipeline does not specify its buffer_size, this value will be used."
        ),
    )
    contrastive_size: Optional[PositiveInt] = Field(
        default=None,
        description="Common contrastive size for SlidingPipeline and NegPoolPipeline.",
    )
    prefetch_factor: PositiveInt = Field(
        ...,
        description="Maximum number of prefetched items, equal to the buffer size of DataGateway",
    )
    batch_size: PositiveInt = Field(..., description="Batch size used during training.")
    seed: Optional[int] = Field(
        default=None, description="Seed value for setup random."
    )
    pipelines: dict[PipelineType, PipelineConfig] = Field(
        ..., description="Dictionary mapping each pipeline type to its config."
    )
    sep_token: str = Field(
        ..., description="Separate token used for cross encoder encoding."
    )
    tokenizer_path: str = Field(..., description="Path to the pretrained tokenizer.")

    @model_validator(mode="before")
    @classmethod
    def validate_pipelines(cls, data: Any):
        data = copy.deepcopy(data)
        if "pipelines" not in data:
            raise ValueError("Missing required field 'pipelines'")
        pipelines_kv = list(data["pipelines"].items())
        for pipeline_type, pipeline_config in pipelines_kv:
            if not pipeline_config:
                data["pipelines"][pipeline_type] = {}
                pipeline_config = data["pipelines"][pipeline_type]
            pipeline_config["type"] = pipeline_type
            if pipeline_type in {
                PipelineType.SLIDING,
                PipelineType.NON_CONCISENESS,
                PipelineType.NON_COHERENCE,
                PipelineType.NON_COVERAGE,
                PipelineType.MODEL_COMPARE,
                PipelineType.NLI_SLIDING,
                PipelineType.NLI_STRATIFIED,
            }:
                if "dataset_path" not in pipeline_config:
                    pipeline_config["dataset_path"] = data["main_data_path"]
                if "buffer_size" not in pipeline_config:
                    pipeline_config["buffer_size"] = data["buffer_size"]
            if pipeline_type in {
                PipelineType.SLIDING,
                PipelineType.NEGPOOL,
                PipelineType.NLI_SLIDING,
            }:
                if "contrastive_size" not in pipeline_config:
                    pipeline_config["contrastive_size"] = data["contrastive_size"]
            if pipeline_type == PipelineType.NEGPOOL:
                if "pos_dataset_path" not in pipeline_config:
                    pipeline_config["pos_dataset_path"] = data["main_data_path"]
                if "neg_dataset_path" not in pipeline_config:
                    pipeline_config["neg_dataset_path"] = data["negpool_data_path"]
                if "buffer_size" not in pipeline_config:
                    pipeline_config["buffer_size"] = data["buffer_size"]
        return data


class EvalDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    main_data_path: str = Field(
        ...,
        description="Path to the eval bytedataset for all pipelines and positive pipeline of NegPoolPipeline.",
    )
    negpool_data_path: Optional[str] = Field(
        default=None,
        description="Path to the eval negative bytedataset for NegPoolPipeline.",
    )
    buffer_size: Optional[int] = Field(
        default=None,
        description=(
            "Common buffer size used for all pipelines. "
            "If a pipeline does not specify its buffer_size, this value will be used."
        ),
    )
    batch_size: int = Field(..., description="Batch size used during evaluation.")
    pipelines: dict[PipelineType, Optional[int]] = Field(
        ..., description="Mapping from pipeline type to its buffer size."
    )
    max_dataloader_len: PositiveInt = 5000

    @model_validator(mode="before")
    @classmethod
    def validate_eval_config(cls, data: Any):
        data = copy.deepcopy(data)
        if "pipelines" not in data:
            raise ValueError("Missing required field 'pipelines'")
        pipelines_kv = list(data["pipelines"].items())
        for pipeline_type, buffer_size in pipelines_kv:
            if not buffer_size:
                data["pipelines"][pipeline_type] = data["buffer_size"]
        return data
