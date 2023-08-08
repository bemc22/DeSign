import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn, variable_scope, math_ops, control_flow_ops, array_ops, state_ops, resource_variable_ops
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer, training_ops
import numpy as np

from binarynet.utils import *

all_layers = []


class Dense_BinaryLayer(tf.keras.layers.Dense):
    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(Dense_BinaryLayer, self).__init__(units=output_dim,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                trainable=trainable,
                                                name=name,
                                                **kwargs)

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        self.W_LR_scale = W_LR_scale

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = tensor_shape.TensorShape(input_shape).as_list()[-1]
        num_units = self.units

        if self.H == "Glorot":
            # weight init method
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        # each layer learning rate
        self.W_LR_scale = np.float32(
            1. / np.sqrt(1.5 / (num_inputs + num_units)))

        self.kernel_initializer = tf.random_uniform_initializer(
            -self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = self.add_weight('binary_weight',
                                        shape=[input_shape[-1], self.units],
                                        initializer=tf.random_uniform_initializer(
                                            -self.H, self.H),
                                        regularizer=None,
                                        constraint=None,
                                        dtype=self.dtype,
                                        trainable=False)

        super(Dense_BinaryLayer, self).build(input_shape)

        #tf.add_to_collection('real', self.trainable_variables)
        tfv1.add_to_collection(self.name + '_binary',
                               self.kernel)  # layer-wise group
        # global group
        tfv1.add_to_collection('binary', self.kernel)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)
        #r_kernel = self.kernel
        #self.kernel = self.b_kernel

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(
                inputs, self.b_kernel, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.b_kernel)

        # restore weight
        #self.kernel = r_kernel

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Functional interface for the Dense_BinaryLayer class.


def dense_binary(
        inputs, units,
        activation=None,
        use_bias=True,
        binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None):

    layer = Dense_BinaryLayer(units,
                              activation=activation,
                              use_bias=use_bias,
                              binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint,
                              trainable=trainable,
                              name=name,
                              dtype=inputs.dtype.base_dtype)

    return layer(inputs)


class Conv2D_BinaryLayer(tf.keras.layers.Conv2D):
    '''
    __init__(): init variable
    conv2d():   Functional interface for the 2D convolution layer.
                This layer creates a convolution kernel that is convolved(actually cross-correlated)
                with the layer input to produce a tensor of outputs.
    apply():    Apply the layer on a input, This simply wraps `self.__call__`
    __call__(): Wraps `call` and will be call build(), applying pre- and post-processing steps
    call():     The logic of the layer lives here
    '''

    def __init__(self, kernel_num,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv2D_BinaryLayer, self).__init__(filters=kernel_num,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 data_format=data_format,
                                                 dilation_rate=dilation_rate,
                                                 activation=activation,
                                                 use_bias=use_bias,
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 bias_regularizer=bias_regularizer,
                                                 activity_regularizer=activity_regularizer,
                                                 kernel_constraint=kernel_constraint,
                                                 bias_constraint=bias_constraint,
                                                 trainable=trainable,
                                                 name=name,
                                                 **kwargs)

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        self.W_LR_scale = W_LR_scale

        all_layers.append(self)

    def build(self, input_shape):
        num_inputs = np.prod(self.kernel_size) * \
            tensor_shape.TensorShape(input_shape).as_list()[3]
        num_units = np.prod(self.kernel_size) * self.filters

        if self.H == "Glorot":
            # weight init method
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
        # each layer learning rate
        self.W_LR_scale = np.float32(
            1. / np.sqrt(1.5 / (num_inputs + num_units)))

        self.kernel_initializer = tf.random_uniform_initializer(
            -self.H, self.H)
        self.kernel_constraint = lambda w: tf.clip_by_value(w, -self.H, self.H)

        self.b_kernel = 0  # add_variable must execute before call build()

        super(Conv2D_BinaryLayer, self).build(input_shape)

        tfv1.add_to_collection(self.name + '_binary',
                               self.kernel)  # layer-wise group
        tfv1.add_to_collection('binary', self.kernel)

    def call(self, inputs):
        # binarization weight
        self.b_kernel = binarization(self.kernel, self.H)

        paddings = tf.constant([
            [0 , 0],
            [1 , 1],
            [1 , 1],
            [0 , 0]
        ])

        inputs = tf.pad(inputs, paddings , "SYMMETRIC")

        # inputs = circular_shift(inputs, batch=True, count=1)
        # kernel = circular_shift(self.b_kernel, batch=False, count=1)

        outputs = self.convolution_op(inputs, self.b_kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] *
                                                    outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(
                        outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Functional interface for the Conv2D_BinaryLayer.


def conv2d_binary(inputs,
                  kernel_num,
                  kernel_size,
                  strides=(1, 1),
                  padding='valid',
                  data_format='channels_last',
                  dilation_rate=(1, 1),
                  activation=None,
                  use_bias=True,
                  binary=True, stochastic=True, H=1., W_LR_scale="Glorot",
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None):

    layer = Conv2D_BinaryLayer(
        kernel_num=kernel_num,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype)
    return layer(inputs)

# Not yet binarized


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=tf.zeros_initializer(),
                 gamma_initializer=tf.ones_initializer(),
                 moving_mean_initializer=tf.zeros_initializer(),
                 moving_variance_initializer=tf.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(BatchNormalization, self).__init__(axis=axis,
                                                 momentum=momentum,
                                                 epsilon=epsilon,
                                                 center=center,
                                                 scale=scale,
                                                 beta_initializer=beta_initializer,
                                                 gamma_initializer=gamma_initializer,
                                                 moving_mean_initializer=moving_mean_initializer,
                                                 moving_variance_initializer=moving_variance_initializer,
                                                 beta_regularizer=beta_regularizer,
                                                 gamma_regularizer=gamma_regularizer,
                                                 beta_constraint=beta_constraint,
                                                 gamma_constraint=gamma_constraint,
                                                 renorm=renorm,
                                                 renorm_clipping=renorm_clipping,
                                                 renorm_momentum=renorm_momentum,
                                                 fused=fused,
                                                 trainable=trainable,
                                                 name=name,
                                                 **kwargs)
        # all_layers.append(self)

    def build(self, input_shape):
        super(BatchNormalization, self).build(input_shape)
        self.W_LR_scale = np.float32(1.)

