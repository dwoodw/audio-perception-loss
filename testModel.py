import tensorflow as tf

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
        
        self.conv1 = Conv2D(32, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.conv2 = Conv2D(64, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.conv3 = Conv2D(128, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.conv4 = Conv2D(256, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.conv5 = Conv2D(512, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.conv6 = Conv2D(1024, [5,5], strides = (2,2), padding = 'same', kernel_initializer= tf.initializers.he_uniform(50) )
        self.batch1 = BatchNormalization(axis = -1)
        self.batch2 = BatchNormalization(axis = -1)
        self.batch3 = BatchNormalization(axis = -1)
        self.batch4 = BatchNormalization(axis = -1)
        self.batch5 = BatchNormalization(axis = -1)
        self.dense1 = Dense(4096, activation='relu')
        self.dense2 = Dense(2048, activation='relu')
        self.dense3 = Dense(1024, activation='relu')
        self.dense4 = Dense(512, activation='relu')
        self.dense5 = Dense(100, activation = 'softmax')

        self.dropout = Dropout(0.2)

        self.flat = Flatten()

        
    def call(self, inputs):
        split1, split2 = tf.split(inputs, 2, 3)
        diff = tf.math.subtract(split1, split2)
        diff_flat = tf.reshape(diff, [tf.shape(diff)[0], tf.shape(diff)[1], tf.shape(diff)[2]])

        diff_flat = tf.keras.layers.Permute((2,1))(diff_flat)
        stfts = tf.signal.stft(diff_flat, frame_length=self.frame_len, frame_step=self.hop, fft_length=self.frame_len, window_fn=tf.signal.hann_window)
        stfts = tf.keras.layers.Permute((1,3,2))(stfts)
        stfts = tf.reverse(stfts, [2])
        mix_mag_o = tf.abs(stfts)
        mix_mag = mix_mag_o[:,:,:self.keepFreqs,:]
        current_layer = mix_mag
        #down layer 1
        c1 = self.conv1(current_layer)
        b1 = self.batch1(c1)
        e1 = tf.keras.activations.elu(b1)
        e1 = self.dropout(e1)
        #down layer 2
        c2 = self.conv2(e1)
        b2 = self.batch2(c2)
        e2 = tf.keras.activations.elu(b2)
        e2 = self.dropout(e2)
        #down layer 3
        c3 = self.conv3(e2)
        b3 = self.batch3(c3)
        e3 = tf.keras.activations.elu(b3)
        e3 = self.dropout(e3)
        #down layer 4
        c4 = self.conv4(e3)
        b4 = self.batch4(c4)
        e4 = tf.keras.activations.elu(b4)
        e4 = self.dropout(e4)
        #down layer 5
        c5 = self.conv5(e4)
        b5 = self.batch5(c5)
        e5 = tf.keras.activations.elu(b5)
        e5 = self.dropout(e5)

        #down layer 6
        c6 = self.flat(e5)

        #Dense layers section
        
        d1 = self.dense1(c6)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        d4 = self.dense4(d3)
        d5 = self.dense5(d4)
        return d5