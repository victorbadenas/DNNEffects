import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
from tensorflow.keras.layers import Layer, Dense, Input, Conv1D, LocallyConnected1D, MaxPooling1D, ZeroPadding1D, Permute, UpSampling1D, Conv1DTranspose
from tensorflow.keras import Model

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

class Encoder(Layer):
    def __init__(self, frame_length, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.conv1 = Conv1D(filters=128, kernel_size=64, padding='same')
        self.abs = tf.keras.backend.abs
        self.conv2 = Conv1D(filters=128, kernel_size=128, padding='same', activation='softplus')
        self.pool = MaxPooling1D(16, frame_length//64, padding='same')

    def call(self, inputTensor):
        R = self.conv1(inputTensor)
        x = self.abs(R)
        x = self.conv2(x)
        x = self.pool(x)
        return x, R

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frame_length': self.frame_length
        })
        return config

class Decoder(Layer):
    def __init__(self, frame_length, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.upsampling = UpSampling1D(size=frame_length//64)
        self.dense1 = Dense(128, activation='softplus')
        self.dense2 = Dense(64, activation='softplus')
        self.dense3 = Dense(64, activation='softplus')
        self.dense4 = Dense(128, activation='relu')
        self.convtranspose1 = Conv1DTranspose(1, 128, padding='same')

    def call(self, inputTensor, R):
        x = self.upsampling(inputTensor)
        x *= R
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.convtranspose1(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frame_length': self.frame_length
        })
        return config


class Bypass(Layer):
    def __init__(self, **kwargs):
        super(Bypass, self).__init__(**kwargs)

    def call(self, inputs, fxs):
        return inputs


class ZFC(Layer):
    def __init__(self, **kwargs):
        super(ZFC, self).__init__(**kwargs)
        self.dense1 = Dense(256, activation='softplus')
        self.dense2 = Dense(128, activation='softplus')
        self.dense3 = Dense(128, activation='softplus')
        self.dense4 = Dense(64, activation='softplus')
        self.permute = Permute((2, 1))

    def call(self, inputs, fxs):
        inputs = tf.concat([inputs, tf.repeat(tf.expand_dims(fxs, 1), 64, 1)], axis=2)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.permute(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.permute(x)
        return x


def createEncoderDecoder(parameters):
    encoder_input = Input(shape=(parameters.frame_length, 1), batch_size=parameters.batch_size, name="frame")
    zdnn_fxs = Input((1,), batch_size=parameters.batch_size, name="fxsetting")

    x, R = Encoder(parameters.frame_length)(encoder_input)
    x = Bypass()(x, zdnn_fxs)
    x = Decoder(parameters.frame_length)(x, R)

    model = Model(inputs=[encoder_input, zdnn_fxs], outputs=[x])
    return model


def createZDNNNetwork(parameters, pretrainedEncoderDecoderPath):
    encoder_input = Input(shape=(parameters.frame_length, 1), batch_size=parameters.batch_size, name="frame")
    zdnn_fxs = Input((1,), batch_size=parameters.batch_size, name="fxsetting")

    encoderDecoder = createEncoderDecoder(parameters)
    encoderDecoder.load_weights(pretrainedEncoderDecoderPath)
    encoder = encoderDecoder.get_layer("encoder")
    decoder = encoderDecoder.get_layer("decoder")
    zdnn = ZFC()

    x, R = encoder(encoder_input)
    x = zdnn(x, zdnn_fxs)
    x = decoder(x, R)

    model = Model(inputs=[encoder_input, zdnn_fxs], outputs=[x])
    model.get_layer("encoder").trainable = False
    model.get_layer("frame").trainable = False
    model.get_layer("fxsetting").trainable = False
    model.get_layer("decoder").trainable = False
    return model


if __name__ == "__main__":
    import numpy as np

    class Parameters:
        def __init__(self):
            self.frame_length = 4096
            self.batch_size = 128
    parameters = Parameters()

    decoderPath = "/opt/DNNEffects/models/CAFx_enc_dec_4096_128/models/decoder.h5"
    encoderPath = "/opt/DNNEffects/models/CAFx_enc_dec_4096_128/models/encoder.h5"

    # autoencoder = createCAFx(parameters, encoderPath, decoderPath)
    autoencoder = createZDNNNetwork(parameters, "models/CAFx_enc_dec_4096_128/models/model.012-0.000008.h5")

    x, x2 = np.random.rand(parameters.batch_size, parameters.frame_length, 1), np.random.rand(parameters.batch_size, 1)
    inputs = dict(frame=x, fxsetting=x2)
    outputs = autoencoder(inputs)

    autoencoder.summary()
