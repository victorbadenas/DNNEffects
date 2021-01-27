import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
from tensorflow.keras.layers import Dense

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.dense1 = Dense(256, activation=nn.relu)
        self.dense2 = Dense(256)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
