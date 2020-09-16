import json
import torch
import logging
from pathlib import Path


class ModelIO:
    def __init__(self, parameters, model):
        self.model = model
        self.parameters = parameters
        self.best_metric = None

    def build_checkpoint_path(self, epoch):
        return Path("experiments", self.parameters.name, f"model_{epoch}.pt")

    def build_config_path(self, epoch):
        return Path("experiments", self.parameters.name, f"model_{epoch}.json")

    def load_model_checkpoint(self, checkpoint_path):
        model_config = self._load_model_config(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.best_metric = model_config['best_metric']
        return model_config['epoch']

    @staticmethod
    def _load_model_config(checkpoint_path):
        checkpoint_config_path = checkpoint_path.replace(".pt", ".json")
        with open(checkpoint_config_path, 'r') as f:
            return json.load(f)

    def save_model(self, epoch, metric):
        if self.best_metric is None:
            logging.info(f"Saving model in epoch {epoch}")
            self.__save(epoch, metric)
        elif metric < self.best_metric:
            logging.info(f"Saving model in epoch {epoch}")
            self.__save(epoch, metric)

    def __save(self, epoch, metric):
        config = self.build_config(epoch, metric)
        checkpoint_path = self.build_checkpoint_path(epoch)
        config_path = self.build_config_path(epoch)
        self.__create_folders(checkpoint_path)
        self.__save_checkpoint(checkpoint_path)
        self.__save_model_config(config_path, config)
        self.__update_best_metric(metric)

    def __update_best_metric(self, metric):
        self.best_metric = metric

    def build_config(self, epoch, metric):
        parameters = {}
        for key, value in self.parameters.__dict__.items():
            parameters[key] = str(value) if isinstance(value, Path) else value
        if isinstance(metric, torch.Tensor):
            config = {"epoch": epoch, "best_metric": metric.item(), "parameters": parameters}
        else:
            config = {"epoch": epoch, "best_metric": metric, "parameters": parameters}
        return config

    def __save_checkpoint(self, checkpoint_path):
        logging.info(f"model saved to {checkpoint_path}")
        with open(checkpoint_path, 'wb') as f:
            torch.save(self.model.state_dict(), f)

    def __save_model_config(self, config_path, config):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def __create_folders(path:Path):
        path.parent.mkdir(parents=True, exist_ok=True)
