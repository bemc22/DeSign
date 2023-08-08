import numpy as np

def get_exp_decay(lr_start, lr_end, epochs):

    k_decay = np.log(lr_end / lr_start) * (1 / epochs)

    def exp_decay(epoch, lr):
        print("actual learning_rate: ", lr)
        return lr_start*np.exp(k_decay*epoch)

    return exp_decay


def get_lineal_decay(lr_start, lr_end, epochs):

    m_decay = ( lr_end - lr_start) / epochs
    b_decay = lr_start

    def lineal_decay(epoch, lr):
        print("actual learning_rate: ", lr)
        return b_decay + m_decay*epoch
    
    return lineal_decay