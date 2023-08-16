import scipy.io as sio
import tensorflow as tf
import math

from tensorflow.keras.layers import Layer, Conv2DTranspose, CenterCrop
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import L1
from tensorflow.keras.activations import relu
from itertools import product
import numpy as np


values = [1, -1]
comb = product(values, repeat=9)
KERNELS = np.array(list(comb))
KERNELS = np.reshape(KERNELS, [3, 3, 1, 1, -1])
H = 512


def index_constraint(x): return tf.clip_by_value(x, 0,  1)


THRESHOLD_PATH = 'threshold_q6.npy'


def get_quantizer(thresh_path=THRESHOLD_PATH):

    ths = np.load(thresh_path)
    print(ths)

    @tf.function
    def quantizer(x):

        value = tf.scalar_mul(0., x)
        # count = 1.

        for low, high in zip(ths[:-1], ths[1:]):
            value += tf.sign(relu(x, max_value=high, threshold=low))
            # l = tf.greater( x , low )
            # h = tf.less(x , high)
            # mask = tf.cast(l, tf.float32)*tf.cast(h, tf.float32)
            # value += count*mask
            # count += 1.

        # backward = relu(x, max_value=ths[-2] )
        # backward = relu(x + 1, max_value= 1 + ths[-2])
        # backward = x
        backward = tf.clip_by_value(x, -1, ths[-2])
        return backward + tf.stop_gradient(value - backward)

    return quantizer


@tf.function
def round_through(x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)


@tf.function
def sign_through(x):
    forward = tf.sign(x)  # forward fucntion - sigma
    backward = tf.clip_by_value(x, -1, 1)  # STE - mu
    return backward + tf.stop_gradient(forward - backward)


class customConv2D(Layer):

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 **params):

        super(customConv2D, self).__init__()

        self.strides = strides
        self.padding = padding
        self.filters = filters

    def build(self, input_shape):

        in_channels = int(input_shape[-1])
        index_shape = (1, 1, in_channels, 1, H)

        self.kernel = tf.constant(KERNELS, dtype=tf.float32)
        self.index_kernel = self.add_weight(name='index_kernel', shape=index_shape,
                                            dtype=tf.float32, trainable=True,
                                            constraint=None)

    def call(self, inputs):

        paddings = tf.constant([
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0]
        ])

        x = tf.pad(inputs, paddings, "SYMMETRIC")
        x = tf.nn.conv2d(x, self.get_binary_weights(), strides=self.strides,
                         padding="VALID")

        return x

    def get_binary_weights(self):

        top = self.filters
        index_kernel = tf.math.top_k(self.index_kernel, k=top, sorted=False)[0]
        # index_kernel = tf.tile(self.index_kernel[..., None], [1, 1, 1, 1, top])
        index_kernel = tf.transpose(index_kernel, perm=[0, 1, 2, 4, 3])
        index_kernel = sign_through(self.index_kernel - index_kernel)
        index_kernel = (index_kernel+1)*(index_kernel-1)*(1)
        # index_kernel = tf.where( tf.equal(index_kernel, 0.0) , 1.0, 0.0)
        # index_kernel = tf.sparse.from_dense(index_kernel)
        kernel = tf.reduce_sum(index_kernel*(self.kernel),
                               axis=-1, keepdims=False)

        return kernel


# KERNEL = [
#     [ 0.4190069841485525, 0.8598138495079375,
#       1.356679662547033 , 1.9984060274039663]
# ]
KERNEL = [
    [1.356679662547033,  0.4190069841485525],
    [1.9984060274039663, 0.8598138495079375],
]

# KERNEL = [
#     [ 0.4190069841485525, 0.8598138495079375,
#       1.356679662547033 , 0.4190069841485525]
# ]

THRESHOLDS = [0,  0.4190069841485525, 0.8598138495079375,
              1.356679662547033, 1.9984060274039663]


class Threshold2D(Layer):

    def __init__(self, filename='./thresholds/baseline2x2.mat',):
        super(Threshold2D, self).__init__()

        self.filename = filename
        if self.filename == "random":
            self.kernel = self.random_kernel((2, 2))
        else:
            self.kernel = self.load_kernel(filename)
            
        self.kernel_size = self.kernel.shape
        self.kernel = tf.constant(self.kernel)[..., None, None]

    def random_kernel(self, shape):
        return tf.random.normal(shape, mean=0.0, stddev=0.05)

    def build(self, input_shape):

        m, n = self.kernel_size
        M, N, _ = input_shape[1:]
        H, W = math.ceil(M/m), math.ceil(N/n)

        self.ones = self.add_weight('ones',
                                    shape=(1, H, W, 1),
                                    dtype=tf.float32,
                                    initializer='ones',
                                    trainable=False)

        self.crop = CenterCrop(M, N)

        self.oshape = tf.constant([1, H*m, W*n, 1])
        self.strides = (m, n)

    def load_kernel(self, filename):
        kernel = sio.loadmat(filename)['kernel']
        kernel = np.float32(kernel)
        return kernel

    def call(self, inputs):

        # spatial_thresh = self.conv2dtranpose(self.ones)
        spatial_thresh = tf.nn.conv2d_transpose(self.ones,
                                                self.kernel,
                                                self.oshape,
                                                strides=self.strides,
                                                padding='VALID')

        spatial_thresh = self.crop(spatial_thresh)

        x = inputs - spatial_thresh
        return x

    def get_config(self):

        config = super().get_config()
        config.update({
            "filename": self.filename,
        })
        return config

class Threshold3D(Layer):

    def __init__(self,
                 filename='./thresholds/threshold_3x3_symmetric_v1.mat',
                 **params):
        super(Threshold3D, self).__init__()

        self.filename = filename
        print("THRESHOLD3D SELECTED, FILENAME: ", filename)
        self.kernel = self.load_kernel(filename)
        self.kernel_size = self.kernel.shape
        self.kernel = tf.constant(self.kernel)[..., None]

    def build(self, input_shape):

        m, n, features = self.kernel_size
        M, N, channels = input_shape[1:]
        H, W = math.ceil(M/m), math.ceil(N/n)

        self.ones = self.add_weight('ones',
                                    shape=(1, H, W, 1),
                                    dtype=tf.float32,
                                    initializer='ones',
                                    trainable=False)

        self.crop = CenterCrop(M, N)

        self.oshape = tf.constant([1, H*m, W*n, features])
        self.replicates = int(channels / features)
        self.strides = (m, n)

    def load_kernel(self, filename):
        kernel = sio.loadmat(filename)['kernel']
        kernel = np.float32(kernel)
        return kernel

    def call(self, inputs):
 

        # spatial_thresh = self.conv2dtranpose(self.ones)
        spatial_thresh = tf.nn.conv2d_transpose(self.ones,
                                                self.kernel,
                                                self.oshape,
                                                strides=self.strides,
                                                padding='VALID')

        spatial_thresh = self.crop(spatial_thresh)
        spatial_thresh = tf.tile(spatial_thresh, [1, 1, 1, self.replicates])

        x = inputs - spatial_thresh
        return x

    def get_config(self):

        config = super().get_config()
        config.update({
            "filename": self.filename,
        })
        return config


class LearnedThreshold2D(Layer):

    def __init__(self):
        super(LearnedThreshold2D, self).__init__()

        self.index_kernel = self.add_weight('kernel', shape=(
            2, 2, 1, 1, 5), dtype=tf.float32, trainable=True)
        self.thresholds = tf.constant(THRESHOLDS)[None, None, None, ...]

    def build(self, input_shape):

        M, N = int(input_shape[1]/2), int(input_shape[2]/2)
        self.ones = self.add_weight('ones', shape=(
            1, M, N, 1), dtype=tf.float32, initializer='ones', trainable=False)
        self.oshape = tf.constant([1, M*2, N*2, 1])

    def get_kernel(self):

        index_kernel = self.index_kernel
        index_kernel = sign_through(tf.reduce_max(
            index_kernel, axis=-1, keepdims=True) - index_kernel)
        index_kernel = (1-index_kernel)*(1)
        kernel = index_kernel.__mul__(self.thresholds)
        kernel = tf.reduce_sum(kernel, axis=-1, keepdims=False)
        return kernel

    def call(self, inputs):

        # spatial_thresh = self.conv2dtranpose(self.ones)
        spatial_thresh = tf.nn.conv2d_transpose(
            self.ones, self.get_kernel(), self.oshape, strides=(2, 2), padding='VALID')
        x = inputs - spatial_thresh
        return x


class RealThreshold2D(Layer):

    def __init__(self):
        super(RealThreshold2D, self).__init__()

        # self.thresholds = tf.constant( THRESHOLDS )[None, None, None, ...]

    def build(self, input_shape):

        M, N, C = int(input_shape[1]/2), int(input_shape[2]/2), input_shape[3]
        self.thresholds = self.add_weight('kernel', shape=(
            2, 2, C, 1), dtype=tf.float32, trainable=True)

        self.ones = self.add_weight('ones', shape=(
            1, M, N, 1), dtype=tf.float32, initializer='ones', trainable=False)
        self.oshape = tf.constant([1, M*2, N*2, C])

    def get_kernel(self):
        return self.thresholds

    def call(self, inputs):

        # spatial_thresh = self.conv2dtranpose(self.ones)
        spatial_thresh = tf.nn.conv2d_transpose(
            self.ones, self.get_kernel(), self.oshape, strides=(2, 2), padding='VALID')
        x = inputs - spatial_thresh
        return x


@tf.function
def ap2(x):
    exponent = tf.round(tf.math.log(tf.abs(x)) / tf.math.log(2.))
    integer = tf.pow(2., exponent)
    return tf.multiply(tf.sign(x), integer)



class customBatchNorm(Layer):

    def __init__(self, *args, **kwargs):
        super(customBatchNorm, self).__init__(*args, **kwargs)
    
    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs) , self.gamma

# class customBatchNorm(Layer):

#     def __init__(self, momentum=0.99, epsilon=0.001):

#         super(customBatchNorm, self).__init__()
#         self.momentum = tf.constant(momentum)
#         self.epsilon = tf.constant(epsilon)

#     def build(self, input_shape):

#         self.dims = len(input_shape) - 1
#         self.axis = list(range(self.dims))

#         param_shape = (1,)*self.dims + (input_shape[-1],)
#         self.moving_mean = self.add_weight(
#             'moving_mean', shape=param_shape, dtype=tf.float32, initializer='zeros', trainable=False)
#         self.moving_var = self.add_weight(
#             'moving_var', shape=param_shape, dtype=tf.float32, initializer='ones', trainable=False)

#         self.gamma = self.add_weight(
#             'gamma', shape=param_shape, dtype=tf.float32, initializer='ones', trainable=True)
#         # self.beta  = self.add_weight('beta', shape=param_shape, dtype=tf.float32, initializer='zeros', trainable=True)

#     def update_variable(self, old_value, new_value):
#         old_value.assign(tf.math.scalar_mul(self.momentum, old_value))
#         old_value.assign_add(tf.math.scalar_mul(1. - self.momentum, new_value))

#     def call(self, inputs, training=None):

#         if training:  # update mean
#             mean = tf.reduce_mean(inputs, axis=self.axis,
#                                   keepdims=True)  # E(x)
#             self.update_variable(self.moving_mean, mean)

#         buffer = tf.add(inputs, - self.moving_mean)  # x - E(x)

#         if training:  # update var
#             var = tf.reduce_mean(tf.multiply(buffer, ap2(
#                 buffer)), axis=self.axis, keepdims=True)  # [ x - E(x) ]^2
#             var = tf.pow(tf.sqrt(var + self.epsilon), -1)
#             self.update_variable(self.moving_var, var)

#         buffer = tf.multiply(buffer, ap2(self.moving_var))
#         buffer = tf.multiply(buffer, ap2(self.gamma))
#         # buffer = tf.add(buffer, self.beta)

#         return buffer
