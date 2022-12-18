import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ELU,
    LeakyReLU,
    Multiply,
    ReLU,
    Flatten,
    Softmax)

class Unet(tf.keras.Model):
    """Unet model as per Spleeter"""
    def __init__(self, model_config):
        super(Unet,self).__init__()
        self.model_config = model_config
        self.source_names = model_config["source_names"]
        self.return_spectrogram = model_config['return_spectrogram']
        self.frame_len= model_config['fft_size']
        self.hop = model_config['hop']
        self.keepFreqs = model_config['keepFreqs']
        
        self.dense1 = Dense(4096, activation='elu')
        self.dense2 = Dense(4096, activation='elu')
        self.dense3 = Dense(1024, activation='elu')
        self.dense4 = Dense(512, activation='elu')
        self.dense5 = Dense(1)

        self.dropout = Dropout(0.2)

        self.flat = Flatten()

        
    def call(self, inputs):
        #Dense layers section
        d1 = self.dense1(inputs)
        dr1= self.dropout(d1)

        d2 = self.dense2(dr1)
        dr2= self.dropout(d2)

        d3 = self.dense3(dr2)
        dr3= self.dropout(d3)

        d4 = self.dense4(dr3)
        dr4= self.dropout(d4)

        d5 = self.dense5(dr4)

        return d5