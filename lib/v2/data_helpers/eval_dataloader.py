from .gateway import DataGateway
from .schemas import DataGatewayConfig, EvalDataConfig


class EvalDataLoader:
    def __init__(self, config: EvalDataConfig, data_gateway_config: DataGatewayConfig):
        self.config = config
        self.data_gateway_config = data_gateway_config

    def __iter__(self):
        data_gateway_config_dict = {
            "main_data_path": self.config.main_data_path,
            "negpool_data_path": self.config.negpool_data_path,
            "regulate_factors": {
                pipeline_type: 1 for pipeline_type in self.config.pipelines
            },
            "buffer_size": self.config.buffer_size,
            "batch_size": self.config.batch_size,
            "pipelines": {pipeline_type: {} for pipeline_type in self.config.pipelines},
            "contrastive_size": self.data_gateway_config.contrastive_size,
            "prefetch_factor": self.data_gateway_config.prefetch_factor,
            "sep_token": self.data_gateway_config.sep_token,
            "tokenizer_path": self.data_gateway_config.tokenizer_path,
            "max_input_len": self.data_gateway_config.max_input_len,
            "seed": self.data_gateway_config.seed,
        }
        data_gateway_config = DataGatewayConfig(**data_gateway_config_dict)
        data_gateway = DataGateway(data_gateway_config)
        with data_gateway.pipeline_context():
            consumed = 0
            for batch in data_gateway:
                consumed += len(batch["items"])
                if consumed > self.config.max_dataloader_len:
                    last_batch_len = len(batch["items"]) - (
                        consumed - self.config.max_dataloader_len
                    )
                    last_batch = {
                        "items": batch["items"][:last_batch_len],
                        "lookup": batch["lookup"],
                    }
                    all_unique_ids = set()
                    for item in last_batch["items"]:
                        all_unique_ids.add(item["unique_id"])
                    all_lookup_ids = set(last_batch["lookup"].keys())
                    for unique_id in all_lookup_ids:
                        if unique_id not in all_unique_ids:
                            last_batch["lookup"].pop(unique_id)
                    if last_batch:
                        yield last_batch
                    return
                yield batch
