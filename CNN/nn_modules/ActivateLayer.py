import theano
import theano.tensor as T
import numpy as np

class ActivateLayer(object):
    def __init__(self, input=None, activation=T.tanh, shape=None, filter_shape=None):
        if activation == None:
            activation = T.tanh

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.output = activation(input + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.b]

        self.shape = shape

