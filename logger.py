import wandb


class LoggerService:
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb

    def initialize(self, model, model_config, model_name):
        if self.use_wandb:
            wandb.init(project='aso', config=model_config, name=model_name)
            wandb.watch(model)

    def log(self, dict_value):
        if self.use_wandb:
            wandb.log(dict_value)
