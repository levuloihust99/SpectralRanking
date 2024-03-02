import wandb

from typing import Text, Dict, Any


class Reporter:
    ...


class WandBReporter(Reporter):
    def __init__(self, api_key, run_id):
        wandb.login(key=api_key)
        self.run_id = run_id

    def init_run(
        self,
        project_name: Text,
        config: Dict[Text, Any]
    ):
        wandb.init(project=project_name, name=self.run_id, config=config)

    def log(self, metrics: Dict[Text, Any], step: int):
        wandb.log(metrics, step=step)


class TensorboardReporter(Reporter):
    ...
