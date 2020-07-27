"""
Copyright (c) 2018 CRISP

Layer with a trainable bias combined with relu activation.

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


class TrainableThresholdRelu(Dense):
    """
    Layer that is just a trainable bias combined with relu activation. Note that
    it is inherited from the Dense layer.
    We are just going to initialize in the build function differently.
    This is shrinkage operator with lambda/L.
    """

    def __init__(self, output_dim, num_conv, L, lambda_trainable, twosided, **kwargs):
        """
        Constructor. Instantiates a Dense layer with relu activation.
        :param output_dim: output dimension
        :param num_conv: number of convolutions
        :param L: 1/L is the step size from ISTA or FISTA
        :param lambda_trainable: set True for trainable threshold
        :param twosided: set True for twosided relu
        :param kwargs: activation must be relu
        """
        self.output_dim = output_dim
        self.num_conv = num_conv
        self.L = L
        self.lambda_trainable = lambda_trainable
        self.twosided = twosided
        if "activation" in kwargs:
            raise ValueError(
                "Cannot overload activation in TrainableThresholdedRelu. Activation is always ReLu."
            )
        super().__init__(output_dim[-1], activation="relu", **kwargs)

    def build(self, input_shape):
        """
        The only function that is overloaded from the Dense layer class.
        We Set bias to be trainable.
        :param input_shape: input tensor shape
        :return: none
        """
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            shape=(self.num_conv,),
            initializer=self.bias_initializer,
            name="bias",
            regularizer=self.bias_regularizer,
            trainable=self.lambda_trainable,
            constraint=non_neg(),
        )

        self.built = True

    def call(self, inputs):
        output = inputs
        if self.use_bias:
            # multiply lambda / L to be the bias
            bias_with_L = self.bias / self.L
            bias_with_L = tf.cast(bias_with_L, tf.float32)
            if len(self.output_dim) == 2:
                bias_vector = tf.add(
                    bias_with_L[0], tf.zeros((self.output_dim[0], 1), dtype=tf.float32)
                )
            elif len(self.output_dim) == 3:
                bias_vector = tf.add(
                    bias_with_L[0],
                    tf.zeros(
                        (self.output_dim[0], self.output_dim[1], 1), dtype=tf.float32
                    ),
                )
            # apply a different bias for each convolution kernel
            for n in range(self.num_conv - 1):
                if len(self.output_dim) == 2:
                    temp = tf.add(
                        bias_with_L[n + 1],
                        tf.zeros((self.output_dim[0], 1), dtype=tf.float32),
                    )
                    bias_vector = tf.concat([bias_vector, temp], axis=1)
                elif len(self.output_dim) == 3:
                    temp = tf.add(
                        bias_with_L[n + 1],
                        tf.zeros(
                            (self.output_dim[0], self.output_dim[1], 1),
                            dtype=tf.float32,
                        ),
                    )
                    bias_vector = tf.concat([bias_vector, temp], axis=2)
            # add bias
            output_pos = K.bias_add(output, -1 * bias_vector)
            if self.twosided:
                output_neg = K.bias_add(output, bias_vector)
        # applt shrinkage
        if self.activation is not None:
            output_pos = self.activation(output_pos)
            if self.twosided:
                output_neg = -1 * self.activation(-1 * output_neg)
            if self.twosided:
                output = output_pos + output_neg
            else:
                output = output_pos
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.num_conv
        return tuple(output_shape)


class TrainableThresholdRelu_learned(Dense):
    """
    Layer that is just a trainable bias combined with relu activation. Note that
    it is inherited from the Dense layer.
    We are just going to initialize in the build function differently.
    This Relu is has trainable threshold
    """

    def __init__(self, output_dim, num_conv, twosided, **kwargs):
        """
        Constructor. Instantiates a Dense layer with relu activation.
        :param output_dim: output dimension
        :param num_conv: number of convolutions
        :param two_sided: set True for two-sided relu
        :param kwargs: activation must be relu
        """
        self.output_dim = output_dim
        self.num_conv = num_conv
        self.twosided = twosided
        if "activation" in kwargs:
            raise ValueError(
                "Cannot overload activation in TrainableThresholdedRelu. Activation is always ReLu."
            )
        super().__init__(output_dim, activation="relu", **kwargs)

    def build(self, input_shape):
        """
        The only function that is overloaded from the Dense layer class.
        We Set bias to be trainable.
        :param input_shape: input tensor shape
        :return: none
        """
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            shape=(self.num_conv,),
            initializer=self.bias_initializer,
            name="bias",
            regularizer=self.bias_regularizer,
            trainable=True,
            constraint=non_neg(),
        )
        self.built = True

    def call(self, inputs):
        output = inputs
        if self.use_bias:
            if len(self.output_dim) == 2:
                bias_vector = tf.add(
                    self.bias[0], tf.zeros((self.output_dim[0], 1), dtype=tf.float32)
                )
            elif len(self.output_dim) == 3:
                bias_vector = tf.add(
                    self.bias[0],
                    tf.zeros(
                        (self.output_dim[0], self.output_dim[1], 1), dtype=tf.float32
                    ),
                )
            # apply a different bias for each convolution kernel
            for n in range(self.num_conv - 1):
                if len(self.output_dim) == 2:
                    temp = tf.add(
                        self.bias[n + 1],
                        tf.zeros((self.output_dim[0], 1), dtype=tf.float32),
                    )
                    bias_vector = tf.concat([bias_vector, temp], axis=1)
                elif len(self.output_dim) == 3:
                    temp = tf.add(
                        self.bias[n + 1],
                        tf.zeros(
                            (self.output_dim[0], self.output_dim[1], 1),
                            dtype=tf.float32,
                        ),
                    )
                    bias_vector = tf.concat([bias_vector, temp], axis=2)
            # add bias
            output_pos = K.bias_add(output, -1 * bias_vector)
            if self.twosided:
                output_neg = K.bias_add(output, bias_vector)
        # applt shrinkage
        if self.activation is not None:
            output_pos = self.activation(output_pos)
            if self.twosided:
                output_neg = -1 * self.activation(-1 * output_neg)
            if self.twosided:
                output = output_pos + output_neg
            else:
                output = output_pos
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.num_conv
        return tuple(output_shape)
