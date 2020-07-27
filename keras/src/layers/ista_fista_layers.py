"""
Copyright (c) 2018 CRISP

Custom layer to do ISTA or FISTA for convolutional dictionary.

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


class ISTA_1d(Conv1D):
    """
    ISTA layer for tied convolutional weights (H,HT) 1d case.
    """

    def __init__(
        self, tied_layer, y, L, lambda_trainable, twosided, num_iterations, **kwargs
    ):
        """
        Constructor
        :param tied_layer: Conv1D layer that used as HT.
        :param y: input data
        :param L: 1/L is the step size from ISTA
        :param lambda_trainable: set True for lambda to be trainable
        :param twosided: set True for twosided relu
        :param num_iterations: max number of iteraitons of ISTA
        :param kwargs:
        """
        self.tied_layer = tied_layer
        # The output dimension is the output dimension of the tied layer
        self.output_dim1 = tied_layer.output_shape[-2]
        self.output_dim2 = tied_layer.output_shape[-1]
        self.kernel_size = self.tied_layer.kernel_size
        self.y = y
        self.L = L
        self.lambda_trainable = lambda_trainable
        self.twosided = twosided
        self.num_iterations = num_iterations
        super().__init__(
            (self.output_dim1, self.output_dim2),
            self.kernel_size,
            activation="relu",
            **kwargs
        )

    def build(self, input_shape):
        # Set the input dimensions as the output dimension of the conv layer
        assert len(input_shape) >= 2
        self.num_data = input_shape[0]
        self.input_dim = self.tied_layer.output_shape[-1]
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: self.input_dim})]
        # Set kernel from the tied layer as flipped
        self.kernel = K.reverse(self.tied_layer.kernel, axes=0)
        self.kernel = K.reshape(
            self.kernel, (self.kernel_size[0], self.tied_layer.output_shape[2], 1)
        )

        # Set bias from the lambda_value
        self.bias = self.add_weight(
            shape=(self.tied_layer.output_shape[2],),
            initializer=self.bias_initializer,
            name="lambda",
            regularizer=self.bias_regularizer,
            trainable=self.lambda_trainable,
            constraint=non_neg(),
        )

        # Have to set build to True
        self.built = True

    def call(self, z):
        def ista_iteration(z_old, ctr):
            """
            ISTA iteration
            :param z_old: sparse code form previous iteraiton
            :param ctr: counter for monitor the iteration
            :return: z_new, ctr+1
            """
            # zero-pad
            paddings = tf.constant(
                [[0, 0], [self.kernel_size[0] - 1, self.kernel_size[0] - 1], [0, 0]]
            )
            z_pad = tf.pad(z_old, paddings, "CONSTANT")
            # Hz
            H_z_old = K.conv1d(z_pad, self.kernel, padding="valid")
            # take residuals
            res = tf.add(self.y, -H_z_old)
            # convolve with HT
            HT_res = K.conv1d(res, self.tied_layer.kernel, padding="valid")
            # divide by L
            HT_res_L = tf.multiply(HT_res, 1 / self.L)
            # get new z before shrinkage
            pre_z_new = tf.add(z_old, HT_res_L)
            # soft-thresholding
            # multiply lambda / L to be the bias
            bias_with_L = self.lambda_value / self.L
            bias_with_L = tf.cast(bias_with_L, tf.float32)
            bias_vector = tf.add(
                bias_with_L[0], tf.zeros((self.output_dim1, 1), dtype=tf.float32)
            )
            # apply a different bias for each convolution kernel
            for n in range(self.output_dim2 - 1):
                temp = tf.add(
                    bias_with_L[n + 1],
                    tf.zeros((self.output_dim1, 1), dtype=tf.float32),
                )
                bias_vector = tf.concat([bias_vector, temp], axis=1)
            # add bias
            output_pos = K.bias_add(pre_z_new, -1 * bias_vector)
            if self.twosided:
                output_neg = K.bias_add(pre_z_new, bias_vector)
            # shrinkage
            output_pos = self.activation(output_pos)
            if self.twosided:
                output_neg = -1 * self.activation(-1 * output_neg)
            if self.twosided:
                output = output_pos + output_neg
            else:
                output = output_pos
            z_new = output
            return z_new, ctr + 1

        def cond(z, ctr):
            """
            condition to monitor the maximum iteraiton
            :param z: sparse code
            :param ctr: counter for monitor the iteration
            :return: boolean, True if ctr < num_iterations
            """
            return tf.less(ctr, self.num_iterations)

        # initialize the while loop variables
        loop_vars = (z, 0)
        # perform ISTA
        output = tf.while_loop(cond, ista_iteration, loop_vars, parallel_iterations=1)
        return output[0]

    def compute_output_shape(self, input_shape):
        # Note output dim was set to the other layers input dim in the constructor
        return input_shape[0], self.output_dim1, self.output_dim2


class FISTA_1d(Conv1D):
    """
    FISTA layer for tied convolutional weights (H,HT) 1d case.
    """

    def __init__(
        self,
        tied_layer,
        y,
        L,
        lambda_trainable,
        twosided,
        num_iterations,
        lambda_single=False,
        lambda_EM=False,
        **kwargs
    ):
        """
        Constructor
        :param tied_layer: Conv1D layer that used as HT.
        :param y: input data
        :param L: 1/L is the step size from FISTA
        :param lambda_trainable: set True for lambda to be trainable
        :param twosided: set True for twosided relu
        :param num_iterations: max number of iteraitons of FISTA
        :param lambda_signel: True for sharing one lambda for all filters
        :param lambda_EM: False for backprop, True for closed-form given code
        :param kwargs:
        """
        self.tied_layer = tied_layer
        # The output dimension is the output dimension of the tied layer
        self.output_dim1 = tied_layer.output_shape[-2]
        self.output_dim2 = tied_layer.output_shape[-1]
        self.kernel_size = self.tied_layer.kernel_size
        self.y = y
        self.L = L
        self.lambda_trainable = lambda_trainable
        self.twosided = twosided
        self.num_iterations = num_iterations
        self.lambda_single = lambda_single
        self.lambda_EM = lambda_EM
        super().__init__(
            (self.output_dim1, self.output_dim2),
            self.kernel_size,
            activation="relu",
            **kwargs
        )

    def build(self, input_shape):
        # Set the input dimensions as the output dimension of the conv layer
        assert len(input_shape) >= 2
        self.num_data = input_shape[0]
        self.input_dim = self.tied_layer.output_shape[-1]
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: self.input_dim})]
        # Set kernel from the tied layer as flipped
        self.kernel = K.reverse(self.tied_layer.kernel, axes=0)
        self.kernel = K.reshape(
            self.kernel, (self.kernel_size[0], self.tied_layer.output_shape[2], 1)
        )

        # Set bias from the lambda_value
        if self.lambda_single:
            self.bias = self.add_weight(
                shape=(1,),
                initializer=self.bias_initializer,
                name="lambda",
                regularizer=self.bias_regularizer,
                trainable=self.lambda_trainable,
                constraint=non_neg(),
            )
        else:
            self.bias = self.add_weight(
                shape=(self.tied_layer.output_shape[2],),
                initializer=self.bias_initializer,
                name="lambda",
                regularizer=self.bias_regularizer,
                trainable=self.lambda_trainable,
                constraint=non_neg(),
            )

        # noiseSTD
        self.noiseSTD = self.add_weight(
            shape=(1,),
            initializer=self.bias_initializer,
            name="noiseSTD",
            regularizer=self.bias_regularizer,
            trainable=False,
            constraint=non_neg(),
        )

        # Have to set build to True
        self.built = True

    def call(self, z):
        def fista_iteration(z_old, x_old, s_old, ctr):
            """
            FISTA iteration
            :param z_old: sparse code form previous iteraiton
            :param x_old: the new point used in FISTA
            :param s_old: s variable used in FISTA
            :param ctr: counter for monitor the iteration
            :return: z_new, x_new, s_new, ctr+1
            """
            s_new = (1.0 + tf.sqrt(1.0 + 4.0 * s_old * s_old)) / 2.0

            # zero-pad
            paddings = tf.constant(
                [[0, 0], [self.kernel_size[0] - 1, self.kernel_size[0] - 1], [0, 0]]
            )
            x_pad = tf.pad(x_old, paddings, "CONSTANT")
            # Hx
            H_x_old = K.conv1d(x_pad, self.kernel, padding="valid")
            # take residuals
            res = tf.add(self.y, -H_x_old)
            # convolve with HT
            HT_res = K.conv1d(res, self.tied_layer.kernel, padding="valid")
            # divide by L
            HT_res_L = tf.multiply(HT_res, 1 / self.L)
            # get new z before shrinkage
            pre_z_new = tf.add(x_old, HT_res_L)
            # soft-thresholding
            # multiply lambda / L to be the bias
            if self.lambda_single:
                bias_with_L = tf.zeros((self.tied_layer.output_shape[2],)) + (
                    (self.bias * (self.noiseSTD ** 2)) / self.L
                )
            else:
                bias_with_L = (self.bias * (self.noiseSTD ** 2)) / self.L
            bias_with_L = tf.cast(bias_with_L, tf.float32)
            bias_vector = tf.add(
                bias_with_L[0], tf.zeros((self.output_dim1, 1), dtype=tf.float32)
            )
            # apply a different bias for each convolution kernel
            for n in range(self.output_dim2 - 1):
                temp = tf.add(
                    bias_with_L[n + 1],
                    tf.zeros((self.output_dim1, 1), dtype=tf.float32),
                )
                bias_vector = tf.concat([bias_vector, temp], axis=1)
            # add bias
            output_pos = K.bias_add(pre_z_new, -1 * bias_vector)
            if self.twosided:
                output_neg = K.bias_add(pre_z_new, bias_vector)
            # shrinkage
            output_pos = self.activation(output_pos)
            if self.twosided:
                output_neg = -1 * self.activation(-1 * output_neg)
            if self.twosided:
                output = output_pos + output_neg
            else:
                output = output_pos
            # get z_new
            z_new = output
            # get x_new point from z_new
            z_new_z_old_res = tf.add(z_new, -1 * z_old)
            t_z_new_z_old_res = tf.multiply(z_new_z_old_res, (s_old - 1.0) / s_new)
            x_new = tf.add(z_new, t_z_new_z_old_res)

            return z_new, x_new, s_new, ctr + 1

        def cond(z, x, s, ctr):
            """
            condition to monitor the maximum iteraiton
            :return: boolean, True if ctr < num_iterations
            """
            return tf.less(ctr, self.num_iterations)

        # initialize s
        s = 1.0
        # initialize the while loop variables
        loop_vars = (z, z, s, 0)
        # perform FISTA
        output = tf.while_loop(cond, fista_iteration, loop_vars, parallel_iterations=1)
        if self.lambda_trainable:
            if not self.lambda_EM:
                lambda_term = tf.zeros((self.output_dim2))
                lambda_term += self.bias
                return [output[0], lambda_term]
            else:
                return output[0]
        else:
            return output[0]

    def compute_output_shape(self, input_shape):
        # Note output dim was set to the other layers input dim in the constructor
        if self.lambda_trainable:
            if not self.lambda_EM:
                if self.lambda_single:
                    return [
                        (input_shape[0], self.output_dim1, self.output_dim2),
                        (input_shape[0], 1),
                    ]
                else:
                    return [
                        (input_shape[0], self.output_dim1, self.output_dim2),
                        (input_shape[0], self.output_dim2),
                    ]
            else:
                return (input_shape[0], self.output_dim1, self.output_dim2)
        else:
            return (input_shape[0], self.output_dim1, self.output_dim2)


class FISTA_2d(Conv2D):
    """
    FISTA layer for tied convolutional weights (H,HT) 2d case.
    """

    def __init__(
        self,
        tied_layer,
        y,
        L,
        lambda_trainable,
        twosided,
        num_iterations,
        lambda_single=False,
        lambda_EM=False,
        **kwargs
    ):
        """
        Constructor
        :param tied_layer: Conv1D layer that used as HT.
        :param y: input data
        :param L: 1/L is the step size from FISTA
        :param lambda_trainable: set True for lambda to be trainable
        :param twosided: set True for twosided relu
        :param num_iterations: max number of iteraitons of FISTA
        :param lambda_signel: True for sharing one lambda for all filters
        :param lambda_EM: False for backprop, True for closed-form given code
        :param kwargs:
        """
        self.tied_layer = tied_layer
        # The output dimension is the output dimension of the tied layer
        self.output_dim1 = tied_layer.output_shape[-3]
        self.output_dim2 = tied_layer.output_shape[-2]
        self.output_dim3 = tied_layer.output_shape[-1]
        self.kernel_size = self.tied_layer.kernel_size
        self.y = y
        self.L = L
        self.lambda_trainable = lambda_trainable
        self.twosided = twosided
        self.num_iterations = num_iterations
        self.lambda_single = lambda_single
        self.lambda_EM = lambda_EM
        super().__init__(
            (self.output_dim1, self.output_dim2, self.output_dim3),
            self.kernel_size,
            activation="relu",
            **kwargs
        )

    def build(self, input_shape):
        # Set the input dimensions as the output dimension of the conv layer
        assert len(input_shape) >= 2
        self.num_data = input_shape[0]
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
                1,
            ),
        )

        # Set bias from the lambda_value
        if self.lambda_single:
            self.bias = self.add_weight(
                shape=(1,),
                initializer=self.bias_initializer,
                name="lambda",
                regularizer=self.bias_regularizer,
                trainable=self.lambda_trainable,
                constraint=non_neg(),
            )
        else:
            self.bias = self.add_weight(
                shape=(self.tied_layer.output_shape[3],),
                initializer=self.bias_initializer,
                name="lambda",
                regularizer=self.bias_regularizer,
                trainable=self.lambda_trainable,
                constraint=non_neg(),
            )

        # noiseSTD
        self.noiseSTD = self.add_weight(
            shape=(1,),
            initializer=self.bias_initializer,
            name="noiseSTD",
            regularizer=self.bias_regularizer,
            trainable=False,
            constraint=non_neg(),
        )

        # Have to set build to True
        self.built = True

    def call(self, z):
        def fista_iteration(z_old, x_old, s_old, ctr):
            """
            FISTA iteration
            :param z_old: sparse code form previous iteraiton
            :param x_old: the new point used in FISTA
            :param s_old: s variable used in FISTA
            :param ctr: counter for monitor the iteration
            :return: z_new, x_new, s_new, ctr+1
            """
            s_new = (1.0 + tf.sqrt(1.0 + 4.0 * s_old * s_old)) / 2.0

            # zero-pad
            paddings = tf.constant(
                [
                    [0, 0],
                    [self.kernel_size[0] - 1, self.kernel_size[0] - 1],
                    [self.kernel_size[1] - 1, self.kernel_size[1] - 1],
                    [0, 0],
                ]
            )
            x_pad = tf.pad(x_old, paddings, "CONSTANT")
            # Hx
            H_x_old = K.conv2d(x_pad, self.kernel, padding="valid")
            # take residuals
            res = tf.add(self.y, -H_x_old)
            # convolve with HT
            HT_res = K.conv2d(res, self.tied_layer.kernel, padding="valid")
            # divide by L
            HT_res_L = tf.multiply(HT_res, 1 / self.L)
            # get new z before shrinkage
            pre_z_new = tf.add(x_old, HT_res_L)
            # soft-thresholding
            # multiply lambda / L to be the bias
            if self.lambda_single:
                bias_with_L = tf.zeros((self.tied_layer.output_shape[3],)) + (
                    (self.bias * (self.noiseSTD ** 2)) / self.L
                )
            else:
                bias_with_L = (self.bias * (self.noiseSTD ** 2)) / self.L
            bias_with_L = tf.cast(bias_with_L, tf.float32)
            bias_vector = tf.add(
                bias_with_L[0],
                tf.zeros((self.output_dim1, self.output_dim2, 1), tf.float32),
            )
            # apply a different bias for each convolution kernel
            for n in range(self.output_dim3 - 1):
                temp = tf.add(
                    bias_with_L[n + 1],
                    tf.zeros((self.output_dim1, self.output_dim2, 1), tf.float32),
                )
                bias_vector = tf.concat([bias_vector, temp], axis=2)
            # add bias
            output_pos = K.bias_add(pre_z_new, -1 * bias_vector)
            if self.twosided:
                output_neg = K.bias_add(pre_z_new, bias_vector)
            # shrinkage
            output_pos = self.activation(output_pos)
            if self.twosided:
                output_neg = -1 * self.activation(-1 * output_neg)
            if self.twosided:
                output = output_pos + output_neg
            else:
                output = output_pos
            # get z_new
            z_new = output
            # get x_new point from z_new
            z_new_z_old_res = tf.add(z_new, -1 * z_old)
            t_z_new_z_old_res = tf.multiply(z_new_z_old_res, (s_old - 1.0) / s_new)
            x_new = tf.add(z_new, t_z_new_z_old_res)

            return z_new, x_new, s_new, ctr + 1

        def cond(z, x, s, ctr):
            """
            condition to monitor the maximum iteraiton
            :return: boolean, True if ctr < num_iterations
            """
            return tf.less(ctr, self.num_iterations)

        # initialize s
        s = 1.0
        # initialize the while loop variables
        loop_vars = (z, z, s, 0)
        # perform FISTA
        output = tf.while_loop(cond, fista_iteration, loop_vars, parallel_iterations=1)

        if self.lambda_trainable:
            if not self.lambda_EM:
                lambda_term = tf.zeros((self.output_dim3))
                lambda_term += self.bias
                return [output[0], lambda_term]
            else:
                return output[0]
        else:
            return output[0]

    def compute_output_shape(self, input_shape):
        # Note output dim was set to the other layers input dim in the constructor
        if self.lambda_trainable:
            if not self.lambda_EM:
                if self.lambda_single:
                    return [
                        (
                            input_shape[0],
                            self.output_dim1,
                            self.output_dim2,
                            self.output_dim3,
                        ),
                        (input_shape[0], 1),
                    ]
                else:
                    return [
                        (
                            input_shape[0],
                            self.output_dim1,
                            self.output_dim2,
                            self.output_dim3,
                        ),
                        (input_shape[0], self.output_dim3),
                    ]
            else:
                return (
                    input_shape[0],
                    self.output_dim1,
                    self.output_dim2,
                    self.output_dim3,
                )
        else:
            return (
                input_shape[0],
                self.output_dim1,
                self.output_dim2,
                self.output_dim3,
            )
