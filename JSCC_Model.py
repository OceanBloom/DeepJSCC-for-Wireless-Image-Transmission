import tensorflow as tf
import numpy as np
import keras

class JSCC_Endcoder(keras.layers.Layer):
    def __init__(self, c, k):
        super(JSCC_Endcoder, self).__init__(name='JSCC_Encoder')
        self.k = k # k是信道带宽 k/n 是压缩比
        self.conv1 = keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same', name='conv1')
        self.prelu1 = keras.layers.PReLU(name='prelu1')
        self.conv2 = keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', name='conv2')
        self.prelu2 = keras.layers.PReLU(name='prelu2')
        self.conv3 = keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', name='conv3')
        self.prelu3 = keras.layers.PReLU(name='prelu3')
        self.conv4 = keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', name='conv4')
        self.prelu4 = keras.layers.PReLU(name='prelu4')
        self.conv5 = keras.layers.Conv2D(filters=c, kernel_size=5, strides=1, padding='same', name='conv5')
        self.prelu5 = keras.layers.PReLU(name='prelu5')
    def Encoder_Norm(self, z):
        tmp = tf.math.sqrt(tf.reduce_sum(tf.math.conj(z) * z))
        tmp2 = tf.complex(tf.math.sqrt(self.k), tf.constant(0.0, dtype=tf.float32))
        return tmp2 * z / tmp # 假设功率P=1
    def call(self, input):
        x = input/255.0
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu2(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.prelu5(x)
        x = tf.reshape(x, (2, -1))
        z = tf.complex(x[0], x[1])
        z = self.Encoder_Norm(z)
        return z
    

class JSCC_Decoder(keras.layers.Layer):
    def __init__(self, c):
        super(JSCC_Decoder, self).__init__(name='JSCC_Decoder')
        self.c = c
        self.trans_conv1 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same', name='trans_conv1')
        self.prelu6 = keras.layers.ReLU(name='prelu6')
        self.trans_conv2 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same', name='trans_conv2')
        self.prelu7 = keras.layers.ReLU(name='prelu7')
        self.trans_conv3 = keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same', name='trans_conv3')
        self.prelu8 = keras.layers.ReLU(name='prelu8')
        self.trans_conv4 = keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same', name='trans_conv4')
        self.prelu9 = keras.layers.ReLU(name='prelu9')
        self.trans_conv5 = keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', name='trans_conv5')
    def call(self, input):
        x = tf.concat([tf.math.real(input), tf.math.imag(input)], 0)
        x = tf.reshape(x, (-1, 8, 8, self.c))
        x = self.trans_conv1(x)
        x = self.prelu6(x)
        x = self.trans_conv2(x)
        x = self.prelu7(x)
        x = self.trans_conv3(x)
        x = self.prelu8(x)
        x = self.trans_conv4(x)
        x = self.prelu9(x)
        x = self.trans_conv5(x)
        x = keras.activations.sigmoid(x)
        x = x*255.0
        return x
        

class JSCC(keras.Model):
    def __init__(self, ratio, SNR):
        super(JSCC, self).__init__(name='JSCC')
        self.k = ratio * 6144.0
        self.c = int( self.k / 64.0)
        self.snr = SNR
        self.noi_pow = 1.0 / (10 ** (self.snr / 10))
        self.encoder = JSCC_Endcoder(self.c, self.k)
        self.decoder = JSCC_Decoder(self.c)
    def call(self, x):
        z = self.encoder(x)
        # self.noise = np.random.normal(0, np.sqrt(self.noi_pow), z.shape)
        real_noise = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=np.sqrt(self.noi_pow / 2))
        imag_noise = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=np.sqrt(self.noi_pow / 2))
        self.comp_noise = tf.complex(real_noise, imag_noise)
        z_hat = z + self.comp_noise
        x_hat = self.decoder(z_hat)
        return x_hat
