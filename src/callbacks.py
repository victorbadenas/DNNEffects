import tensorflow as tf
import logging

class CustomLogging(tf.keras.callbacks.Callback):
    def __init__(self, logger=None, bestLoss=None):
        self.logger = logging.getLogger() if logger is None else logger
        self.bestLoss = bestLoss

    def on_train_begin(self, logs=None):
        self.logger.info(" Begin Training ".center(80, '-'))

    def on_train_end(self, logs=None):
        self.logger.info(f"Best model trained with {self.bestLoss}")

    def on_predict_end(self, epoch, logs=None):
        # self.logger.info("on_predict_end")
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self.bestLoss is None:
            self.bestLoss = 'None'
        self.logger.info(f"Epoch {epoch}: loss = {logs['loss']}, val_loss = {logs['val_loss']}, \nval_loss improvement: {self.bestLoss} -> {logs['val_loss']}")
        if self.bestLoss is 'None' or logs['val_loss'] < self.bestLoss:
            self.bestLoss = logs['val_loss']
