import torch
import logging
from model import Model
from model_io import ModelIO
from dataset import LstDataset


class Trainer():
    def __init__(self, parameters):
        self.parameters = parameters
        self.init_model()
        self.init_torch_modules()
        self.init_dataset()

    def init_model(self):
        self.model = Model(self.parameters)
        logging.info(self.model)
        self.modelIO = ModelIO(self.parameters, self.model)
        self.start_epoch = 0
        if self.parameters.checkpoint is not None:
            logging.info(f"loading checkpoint from path {self.parameters.checkpoint}")
            self.start_epoch = self.modelIO.load_model_checkpoint(self.parameters.checkpoint)
            logging.info(f"checkpoint loaded in epoch {self.start_epoch}")

    def init_torch_modules(self):
        self.optimizer = None

    def init_dataset(self):
        self.trainDataset = LstDataset(self.parameters)
        self.testDataset = LstDataset(self.parameters)

    def __call__(self, *args, **kwargs):
        self.train_model(*args, **kwargs)

    def train_model(self):
        logging.info(f"Stating training from epoch {self.start_epoch}")
        for epochIdx in range(self.start_epoch, self.parameters.epochs):
            logging.info(f"Epoch: {epochIdx}")
            logging.info("Train stage")
            train_accuracy = self.train_epoch()
            logging.info("Test stage")
            test_accuracy = self.test_epoch()
            self.modelIO.save_model(epochIdx, test_accuracy)

    def train_epoch(self):
        return 0.0

    def test_epoch(self):
        return 0.0