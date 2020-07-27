"""
Copyright (c) 2018 CRISP

Custom layer to do flip of another conv layer.

:author: Bahareh Tolooshams
"""

from keras import backend as K
from keras.layers import Conv1D, Conv2D, InputSpec, Dense, Reshape
from keras.initializers import (
    Identity,
    Initializer,
    Constant,
    Ones,
    Zeros,
    RandomNormal,
)
from keras.activations import relu
from keras.constraints import non_neg
import numpy as np
import tensorflow as tf


class Conv1DFlip(Conv1D):
    """
    Conv1D layer whose weights are tied (through a flip) to another conv1D layer.
    """

    def __init__(self, tied_layer, **kwargs):
        """
        Constructor
        :param tied_layer: Conv1D layer that shares weights with this layer.
        :param kwargs:
        """
        self.tied_layer = tied_layer
        # The output dimension is the input dimension of the tied layer
        self.output_dim1 = tied_layer.input_shape[-2]
        self.output_dim2 = tied_layer.input_shape[-1]
        self.kernel_size = self.tied_layer.kernel_size
        super().__init__(
            (self.output_dim1, self.output_dim2), self.kernel_size, **kwargs
        )

    def build(self, input_shape):
        """
        Builds Tensorflow DAG for this layer.
        :param input_shape: input tensor shape
        :return: none
        """
        # Set the input dimensions as the output dimension of the conv layer
        assert len(input_shape) >= 2
        self.input_dim = self.tied_layer.output_shape[-1]
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: self.input_dim})]

        # Set kernel from the tied layer as flipped
        self.kernel = K.reverse(self.tied_layer.kernel, axes=0)

        self.kernel = K.reshape(
            self.kernel,
            (
                self.kernel_size[0],
                self.tied_layer.output_shape[2],
                self.tied_layer.input_shape[-1],
            ),
        )

        # Set bias from the tied layer
        if self.tied_layer.use_bias is True:
            self.bias = self.tied_layer.bias
        else:
            self.bias = None

        # Have to set build to True
        self.built = True

    def call(self, x):
        """
        Main compute function. If bias is used, subtract before multiplying with
        kernel. Otw, just multiply by kernel.
        :param x: input tensor
        :return: either K*x or K*(x-b)
        """
        if self.tied_layer.use_bias is True:
            output = K.bias_add(x, -1 * self.bias)
            output = K.conv1d(output, self.kernel)
        else:
            output = K.conv1d(x, self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        """
        Shape of output.
        :param input_shape:
        :return:
        """
        # Note output dim was set to the other layers input dim in the constructor
        return input_shape[0], self.output_dim1, self.output_dim2


class Conv2DFlip(Conv2D):
    """
    Conv2D layer whose weights are tied (through a flip) to another conv2D layer.
    """

    def __init__(self, tied_layer, **kwargs):
        """
        Constructor
        :param tied_layer: Conv2D layer that shares weights with this layer.
        :param kwargs:
        """
        self.tied_layer = tied_layer
        # The output dimension is the input dimension of the tied layer
        self.output_dim1 = tied_layer.input_shape[-3]
        self.output_dim2 = tied_layer.input_shape[-2]
        self.output_dim3 = tied_layer.input_shape[-1]
        self.kernel_size = self.tied_layer.kernel_size
        super().__init__(
            (self.output_dim1, self.output_dim2, self.output_dim3),
            self.kernel_size,
            **kwargs
        )

    def build(self, input_shape):
        """
        Builds Tensorflow DAG for this layer.
        :param input_shape: input tensor shape
        :return: none
        """
        # Set the input dimensions as the output dimension of the conv layer
        assert len(input_shape) >= 2
        # self.input_dim = self.tied_layer.output_shape
        # self.input_spec = [InputSpec(shape=self.input_dim)]
        self.input_dim = self.tied_layer.output_shape[-1]
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: self.input_dim})]

        # Set kernel from the tied layer
        self.kernel = K.reverse(self.tied_layer.kernel, axes=0)
        self.kernel = K.reverse(self.kernel, axes=1)
        self.kernel = K.reshape(
            self.kernel,
            (
                self.kernel_size[0],
                self.kernel_size[1],
                self.tied_layer.output_shape[-1],
                self.tied_layer.input_shape[-1],
            ),
        )

        # Set bias from the tied layer
        if self.tied_layer.use_bias is True:
            self.bias = self.tied_layer.bias
        else:
            self.bias = None

        # Have to set build to True
        self.built = True

    def call(self, x):
        """
        Main compute function. If bias is used, subtract before multiplying with kernel.
        Otw, just multiply by kernel.
        :param x: input tensor
        :return: either K*x or K*(x-b)
        """
        if self.tied_layer.use_bias is True:
            output = K.bias_add(x, -1 * self.bias)
            output = K.conv2d(output, self.kernel)
        else:
            output = K.conv2d(x, self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        """
        Shape of output.
        :param input_shape:
        :return:
        """
        # Note output dim was set to the other layers input dim in the constructor
        return input_shape[0], self.output_dim1, self.output_dim2, self.output_dim3
