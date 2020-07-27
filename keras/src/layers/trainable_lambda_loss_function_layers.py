"""
Copyright (c) 2018 CRISP

Layer with a trainable scalar multiplication.

:author: Bahareh Tolooshams
"""

from keras import backend as K
from keras.layers import Layer
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


class TrainableLambdaLossFunction(Layer):
    """
    Layer that outputs scalar * input - p log(scalar)
    where lambda is a traiable vector.
    """

    def __init__(self, Ne, num_conv, mean_gamma, delta, lambda_single, **kwargs):
        """
        Constructor. Instantiates a layer with trainable scalar called lambda.
        """
        self.Ne = Ne
        self.num_conv = num_conv
        self.mean_gamma = mean_gamma
        self.delta = delta
        self.lambda_single = lambda_single
        # super(Scalar, self).__init__(**kwargs)
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        The only function that is overloaded from the layer class.
        We Set bias to be trainable.
        :param input_shape: input tensor shape
        :return: none
        """
        if self.lambda_single:
            self.scalar = self.add_weight(
                shape=(1,),
                name="lambda",
                initializer="ones",
                dtype="float32",
                trainable=True,
                constraint=non_neg(),
            )
        else:
            self.scalar = self.add_weight(
                shape=(self.num_conv,),
                name="lambda",
                initializer="ones",
                dtype="float32",
                trainable=True,
                constraint=non_neg(),
            )

        self.built = True

    def call(self, inputs):

        # output = K.sum(self.scalar * inputs - self.Ne * K.log(self.scalar), axis=-1)

        delta = self.delta  # rate
        r = delta * self.mean_gamma  # shape
        # gamma_prior_loss = 2 * (r - 1) * K.log(self.scalar) - delta * K.square(
        #     self.scalar
        # )

        if self.lambda_single:
            scalar = np.zeros((self.num_conv,)) + self.scalar
        else:
            scalar = self.scalar

        gamma_prior_loss = (r - 1) * K.log(scalar) - delta * scalar

        output = K.sum(
            scalar * inputs - self.Ne * K.log(scalar) - gamma_prior_loss, axis=-1
        )

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = [(input_shape[0], 1)]
        return output_shape
