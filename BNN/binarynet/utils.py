import tensorflow as tf


def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round_through(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

# The weights' binarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)
def binarization(W, H, binary=True, deterministic=False, stochastic=False, srng=None):
    dim = W.get_shape().as_list()

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W

    else:
        # [-1,1] -> [0,1]
        #Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)

        # Stochastic BinaryConnect
        '''
        if stochastic:
            # print("stoch")
            Wb = tf.cast(srng.binomial(n=1, p=Wb, size=tf.shape(Wb)), tf.float32)
        '''

        # Deterministic BinaryConnect (round to nearest)
        #else:
        # print("det")
        #Wb = tf.round(Wb)

        # 0 or 1 -> -1 or 1
        #Wb = tf.where(tf.equal(Wb, 1.0), tf.ones_like(W), -tf.ones_like(W))  # cant differential
        Wb = H * binary_tanh_unit(W / H)

    return Wb

@tf.function
def circular_shift(inputs , batch=False, count=1):

    if batch:
        shift   = [0, 0, count, 0] 
    else:
        shift   = [0, count, 0, 0]

    x_shift = tf.roll( inputs, shift=shift, axis=[0, 1, 2, 3] )
    xnor = tf.multiply(inputs, x_shift)
    xnor = (xnor + 1)/2
    output = tf.multiply(inputs, xnor)
    return output
