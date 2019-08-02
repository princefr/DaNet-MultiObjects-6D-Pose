from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
import numpy as np


class Conv2DNormalization(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Conv2DNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))


    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output = output * self.gamma
        return output
