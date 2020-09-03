import os
import sys
import torch
import logging
from model import Model
from model_io import ModelIO


class Trainer():
    def __init__(self, parameters):
        self.parameters = parameters
        self.init_model()

    def init_model(self):
        self.model = Model(self.parameters)
        logging.info(self.model)
        self.modelIO = ModelIO(self.parameters, self.model)
        self.epoch = 0
        if self.parameters.checkpoint is not None:
            logging.info(f"loading checkpoint from path {self.parameters.checkpoint}")
            self.epoch = self.modelIO.load_model_checkpoint(self.parameters.checkpoint)
            logging.info(f"checkpoint loaded in epoch {self.epoch}")

    def __call__(self, *args, **kwargs):
        self.train_model(*args, **kwargs)

    def train_model(self):
        pass
