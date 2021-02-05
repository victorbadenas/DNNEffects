import tensorflow.keras as keras
import tensorflow as tf
from dataset import DataGenerator
from model import createEncoderDecoder,createZDNNNetwork
from callbacks import CustomLogging, CustomSaver
import logging
from pathlib import Path
import pickle
import json
import re


class Trainer:
    def __init__(self, parameters):
        self.parameters = parameters
        logging.info("Available Devices:")
        logging.info(tf.config.list_physical_devices('GPU'))

    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)

    def train(self):
        self.__createDatasets()
        self.__createModel()
        self.__createCallBacks()
        self.__trainModel()

    def __createDatasets(self):
        if self.parameters.pretrained is None:
            logging.info("Loading dataset with target same source and target files")
            target_label = 'source'
        else:
            logging.info("Loading dataset with target source and target files")
            target_label = 'target'

        self.trainGenerator = DataGenerator(self.parameters.train_lst,
                                            batch_size=self.parameters.batch_size,
                                            frame_length=self.parameters.frame_length,
                                            target_label=target_label)
        self.testGenerator = DataGenerator(self.parameters.test_lst,
                                            batch_size=self.parameters.batch_size,
                                            frame_length=self.parameters.frame_length,
                                            target_label=target_label)

    def __createModel(self):
        if self.parameters.pretrained is None:
            logging.info("loading encoder decoder with bypass zdnn")
            self.model = createEncoderDecoder(self.parameters)
        else:
            logging.info("loading full zdnn")
            self.model = createZDNNNetwork(self.parameters, self.parameters.pretrained)

        self.model.compile(
            optimizer="Adam", loss="mse", metrics=["mae"]
        )
        summary = self.getSummary()
        logging.info(summary)

    def getSummary(self):
        summary = []
        self.model.summary(line_length=100, print_fn=lambda x: summary.append(x))
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                layer.model().summary(line_length=100, print_fn=lambda x: summary.append(x))
        return "Summary:\n"+'\n'.join(summary)

    def __createCallBacks(self):
        self.modelFilePath = Path('models/' + self.parameters.name + '/models/model.{epoch:03d}-{val_loss:.6f}.h5')
        self.modelFilePath.parent.mkdir(parents=True, exist_ok=True)
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.modelFilePath, save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=f'./runs/{self.parameters.name}'),
            CustomLogging()
        ]

    def __trainModel(self):
        with open(self.modelFilePath.parent.parent / "model.json", "w") as f:
            config = dict()
            for k, v in self.parameters.__dict__.items():
                config[k] = v if not isinstance(v, Path) else str(v)
            json.dump(config, f)

        if self.parameters.checkpoint is None:
            initial_epoch = 0
        else:
            self.model.load_weights(self.parameters.checkpoint)
            initial_epoch = int(re.search('[a-z]*.([0-9]*)-[0-9]*.[0-9]*.h5', self.parameters.checkpoint.name).group(1))

        history = self.model.fit(x=self.trainGenerator,
                       epochs=self.parameters.epochs,
                       validation_data=self.testGenerator,
                       shuffle=False,
                       callbacks=self.callbacks,
                       initial_epoch=initial_epoch)

        with open(self.modelFilePath.parent.parent / "history.pkl", "wb") as f:
            pickle.dump(history.history, f)

    def saveBestModels(self):
        submodel_save_pattern = str(self.modelFilePath.parent / "{0}.h5")
        for layer in self.model.layers:
            if isinstance(layer, keras.Model):
                layer.save_weights(submodel_save_pattern.format(layer.name))
