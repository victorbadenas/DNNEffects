import tensorflow.keras as keras
import tensorflow as tf
from dataset import DataGenerator
from model import TestModel
from callbacks import CustomLogging

class Trainer:
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)

    def train(self):
        self.__createDatasets()
        self.__createModel()
        self.__createCallBacks()
        self.__trainModel()

    def __createDatasets(self):
        self.trainGenerator = DataGenerator(self.parameters.train_lst, batch_size=self.parameters.batch_size, frame_length=self.parameters.frame_length)
        self.testGenerator = DataGenerator(self.parameters.test_lst, batch_size=self.parameters.batch_size, frame_length=self.parameters.frame_length)

    def __createModel(self):
        self.model = TestModel()
        self.model.compile(
            loss='mse'
        )

    def __createCallBacks(self):
        modelFilePath = 'models/' + self.parameters.name + '/model.{epoch:02d}-{val_loss:.2f}.h5'
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=modelFilePath),
            CustomLogging()
        ]

    def __trainModel(self):
        self.model.fit(x=self.trainGenerator,
                       epochs=self.parameters.epochs,
                       validation_data=self.testGenerator,
                       shuffle=False,
                       callbacks=self.callbacks)
