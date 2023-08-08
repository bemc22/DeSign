import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, BatchNormalization, Lambda
from tensorflow.keras.activations import softmax


from binarynet.utils import binary_tanh_unit
import binarynet.layers as binary_layer


H = 1.
W_LR_scale = "Glorot"

BATCH_LAYER = BatchNormalization

print("QUANTIZER FUNCTION LOADED")

q_fun = tf.identity
# q_fun = get_quantizer()
# q_fun = tf.nn.relu
quantizer_fun = Lambda( lambda x: q_fun(x) )

CENTER = True    
SCALE = True

def conv_bn(pre_layer,
            kernel_num,
            kernel_size,
            padding, activation,
            epsilon=1e-4,
            alpha=1 - .125,
            binary=True,
            stochastic=False,
            H=1.,
            W_LR_scale="Glorot",
            threshold_layer = None,
            kernel_filename = None,
            ):

    conv = binary_layer.conv2d_binary(pre_layer,
                                      kernel_num,
                                      kernel_size,
                                      padding=padding,
                                      binary=binary,
                                      stochastic=stochastic,
                                      H=H,
                                      W_LR_scale=W_LR_scale)

    bn = BATCH_LAYER(epsilon=epsilon, momentum=alpha, center=CENTER, scale=SCALE)(conv)

    if threshold_layer:
        bn = threshold_layer(kernel_filename)(bn)
    # bn = quantizer_fun(bn)
    output = activation(bn)
    return output


def conv_pool_bn(pre_layer,
                 kernel_num,
                 kernel_size,
                 padding,
                 pool_size,
                 activation,
                 epsilon=1e-4,
                 alpha=1 - .125,
                 binary=True,
                 stochastic=False,
                 H=1.,
                 W_LR_scale="Glorot",
                 threshold_layer = None,
                 kernel_filename = None):

    conv = binary_layer.conv2d_binary(pre_layer,
                                      kernel_num,
                                      kernel_size,
                                      padding=padding,
                                      binary=binary,
                                      stochastic=stochastic,
                                      H=H,
                                      W_LR_scale=W_LR_scale)

    pool = MaxPooling2D(pool_size=pool_size, strides=pool_size)(conv)
    bn = BATCH_LAYER(epsilon=epsilon, momentum=alpha, center=CENTER , scale=SCALE)(pool)

    if threshold_layer:
        bn = threshold_layer(kernel_filename)(bn)
    # bn = quantizer_fun(bn)
    output = activation(bn)
    return output


def fully_connect_bn(pre_layer,
                     output_dim,
                     act,
                     use_bias,
                     epsilon=1e-4,
                     alpha=1 - .125,
                     binary=True,
                     stochastic=False,
                     H=1.,
                     W_LR_scale="Glorot"):

    pre_act = binary_layer.dense_binary(pre_layer,
                                        output_dim,
                                        use_bias=use_bias,
                                        kernel_constraint=lambda w: tf.clip_by_value(w, -H, H))

    bn = BATCH_LAYER(epsilon=epsilon, momentum=alpha, center=True , scale=True)(pre_act)
    
    if act == None:
        output = bn
    else:
        # bn = quantizer_fun(bn)
        # bn = Threshold2D()(bn)
        output = act(bn)

    return output


def build_model(training=True, num_classes=10, activation=binary_tanh_unit, threshold_layer=None, kernel_filename='./thresholds/baseline2x2.mat', size=32):

    print("LOG_SOFTMAX ENABLE")
    
    conv_params_head = dict(
        padding='valid',
        activation=activation,
        threshold_layer = threshold_layer,
        kernel_filename=kernel_filename,
    )

    conv_params_tail = dict(
        padding='valid',
        activation=activation,
        threshold_layer = threshold_layer,
        kernel_filename=kernel_filename,
    )

    _input = Input((size, size, 3))

    cnn = conv_bn(_input, 128, (3, 3), **conv_params_head)
    cnn = conv_pool_bn(cnn, 128, (3, 3),  pool_size=(2, 2), **conv_params_head)

    cnn = conv_bn(cnn, 256, (3, 3),  **conv_params_head)
    cnn = conv_pool_bn(cnn, 256, (3, 3), pool_size=(2,2), **conv_params_tail)

    cnn = conv_bn(cnn, 512, (3, 3),  **conv_params_tail)
    cnn = conv_pool_bn(cnn, 512, (3, 3), pool_size=(2,2), **conv_params_tail)

    cnn = tf.keras.layers.Flatten()(cnn)

    cnn = fully_connect_bn(cnn, 1024, act=activation, use_bias=True)
    cnn = fully_connect_bn(cnn, 1024, act=activation, use_bias=True)

    _output = fully_connect_bn(cnn, num_classes, act=tf.nn.log_softmax, use_bias=True)


    return Model(_input, _output, name='BinaryNet')
