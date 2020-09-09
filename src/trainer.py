import torch
import logging
import torch.optim as optim
from torch.utils.data import DataLoader
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
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def init_dataset(self):
        self.train_dataset = DataLoader(LstDataset(self.parameters, self.parameters.train_lst), batch_size=self.parameters.batch_size, shuffle=True, drop_last=True)
        self.test_dataset = DataLoader(LstDataset(self.parameters, self.parameters.test_lst), batch_size=self.parameters.batch_size, shuffle=True, drop_last=True)

    def __call__(self, *args, **kwargs):
        self.train_model(*args, **kwargs)

    def train_model(self):
        logging.info(f"Stating training from epoch {self.start_epoch}")
        for epochIdx in range(self.start_epoch, self.parameters.epochs):
            logging.info(f"Epoch: {epochIdx}")
            train_accuracy = self.train_epoch()
            test_accuracy = self.test_epoch()
            self.modelIO.save_model(epochIdx, test_accuracy)

    @staticmethod
    def log_progress(currentidx, length, log_interval, metric):
        if (currentidx + 1) % log_interval == 0 or (currentidx + 1 == length):
            logging.info(f"Progress: {currentidx+1}/{length} batches: MSEError={metric/currentidx:.6f}")

    @staticmethod
    def move_tensors_to_cuda(*args):
        return_args = ()
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return_args += (arg.cuda(),)
        return return_args

    def train_epoch(self):
        self.model.train()
        if self.device == "cuda":
            self.model.cuda()
        logging.info("Train stage")
        train_mse_error = 0.0
        for batch_idx, (source_tensor, target_tensor) in enumerate(self.train_dataset):
            self.log_progress(batch_idx, len(self.train_dataset), self.parameters.log_interval, train_mse_error)
            if self.device == "cuda":
                source_tensor, target_tensor = self.move_tensors_to_cuda(source_tensor, target_tensor)
            self.optimizer.zero_grad()
            outputs = self.model(source_tensor)
            loss = self.criterion(outputs, target_tensor)
            loss.backward()
            train_mse_error += loss.item()/abs(target_tensor).max()**2/2
            self.optimizer.step()
        return train_mse_error / len(self.train_dataset)

    def test_epoch(self):
        logging.info("Test stage")
        self.model.eval()
        if self.device == "cuda":
            self.model.cuda()
        with torch.no_grad():
            test_mse_error = 0.0
            for batch_idx, (source_tensor, target_tensor) in enumerate(self.test_dataset):
                self.log_progress(batch_idx, len(self.test_dataset), self.parameters.log_interval, test_mse_error)
                if self.device == "cuda":
                    source_tensor, target_tensor = self.move_tensors_to_cuda(source_tensor, target_tensor)
                outputs = self.model(source_tensor)
                loss = self.criterion(outputs, target_tensor)
                test_mse_error += loss.item()/abs(target_tensor).max()**2/2
        return test_mse_error / len(self.train_dataset)
