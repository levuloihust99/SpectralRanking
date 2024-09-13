from .gateway import DataGateway
from .schemas import DataGatewayConfig, EvalDataConfig


class EvalDataLoader:
    def __init__(self, config: EvalDataConfig):
        self.config = config

    def __iter__(self):
        data_gateway_config_dict = {
            "main_data_path": self.config.main_data_path,
            "negpool_data_path": self.config.negpool_data_path,
            "regulate_factors": {
                pipeline_type: 1 for pipeline_type in self.config.pipelines
            },
            "buffer_size": self.config.buffer_size,
            "contrastive_size": self.config.buffer_size,
            "prefetch_factor": 1000,
            "batch_size": self.config.batch_size,
            "pipelines": {pipeline_type: {} for pipeline_type in self.config.pipelines},
            "seed": self.config.seed,
        }
        data_gateway_config = DataGatewayConfig(**data_gateway_config_dict)
        data_gateway = DataGateway(data_gateway_config)
        data_gateway.start_worker()
        consumed = 0
        for batch in data_gateway:
            consumed += len(batch)
            if consumed > self.config.max_dataloader_len:
                last_batch_len = len(batch) - (
                    consumed - self.config.max_dataloader_len
                )
                last_batch = batch[:last_batch_len]
                if last_batch:
                    yield last_batch
                return
            yield batch
