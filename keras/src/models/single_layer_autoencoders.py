"""
Copyright (c) 2019 CRISP

Single layer convolutional autoencoders.

:author: Bahareh Tolooshams
"""

from keras.models import Model
from keras.layers import Conv1D, Conv2D, Input, add, Dense, Flatten, Reshape
from keras.layers import ZeroPadding1D, ZeroPadding2D, Lambda
from keras.initializers import Identity
from keras import regularizers, initializers
from src.layers.conv_tied_layers import Conv1DFlip, Conv1DFlip
from src.layers.trainable_threshold_relu_layers import (
    TrainableThresholdRelu,
    TrainableThresholdRelu_learned,
)
from keras.constraints import non_neg


def decoder_1d(Ne, dictionary_dim, num_conv):
    """
    Create decoder for conv 1d
    :param Ne: dimesion of code
    :param dictionary_dim: dimension of dictionary
    :param num_conv: number of conv filters
    :return: decoder
    """
    # input
    input_signal = Input(shape=(Ne, num_conv), name="input")  # Input placeholder
    # Zero-pad
    input_signal_padded = ZeroPadding1D(padding=(dictionary_dim - 1), name="zeropad")(
        input_signal
    )
    # build convolution
    decoded = Conv1D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=False,
        input_shape=(Ne, num_conv),
        name="decoder",
    )(input_signal_padded)
    # output Y = HZ
    decoder = Model(input_signal, decoded)
    return decoder


def create_unconstrained_linear_autoencoder_1d(input_dim, dictionary_dim, num_conv, L):
    """
    Create a single hidden layer, purely 1d linear auto-encoder.
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :param L:
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(shape=(input_dim, 1), name="input")  # Input placeholder

    # Build encoder
    encoder_layer = Conv1D(
        filters=num_conv,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        input_shape=(input_dim, 1),
        name="encoder",
    )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding1D(padding=(dictionary_dim - 1), name="zeropad")(encoded)

    # Build decoder
    decoder_layer = Conv1D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        name="decoder",
    )
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Divided by L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_unconstrained_linear_autoencoder_2d(input_dim, dictionary_dim, num_conv, L):
    """
    Create a single hidden layer, purely 1d linear auto-encoder.
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :param L:
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(
        shape=(input_dim[0], input_dim[1], 1), name="input"
    )  # Input placeholder

    # Build encoder
    encoder_layer = Conv2D(
        filters=num_conv,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        input_shape=(input_dim[0], input_dim[1], 1),
        name="encoder",
    )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding2D(
        padding=(
            (dictionary_dim[0] - 1, dictionary_dim[0] - 1),
            (dictionary_dim[1] - 1, dictionary_dim[1] - 1),
        ),
        name="zeropad",
    )(encoded)

    # Build decoder
    decoder_layer = Conv2D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        name="decoder",
    )
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Mupltiply by 1/L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_unconstrained_nonlinear_autoencoder_1d(
    input_dim, dictionary_dim, num_conv, L, l1_regularize=False
):
    """
    Create a single hidden layer autoencoder with relu after encoder.
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :param L:
    :param l1_regularize: set to True if want to regularize the output of the encoder
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(shape=(input_dim, 1), name="input")  # Input placeholder

    # Build encoder
    if l1_regularize:
        encoder_layer = Conv1D(
            filters=num_conv,
            kernel_size=dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            activity_regularizer=regularizers.l1(0.00001),
            input_shape=(input_dim, 1),
            name="encoder",
        )

    else:
        encoder_layer = Conv1D(
            filters=num_conv,
            kernel_size=dictionary_dim,
            padding="valid",
            use_bias=False,
            activation="relu",
            trainable=True,
            input_shape=(input_dim, 1),
            name="encoder",
        )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding1D(padding=(dictionary_dim - 1), name="zeropad")(encoded)

    # Build decoder
    decoder_layer = Conv1D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        name="decoder",
    )
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Divided by L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_unconstrained_nonlinear_autoencoder_2d(
    input_dim, dictionary_dim, num_conv, L, l1_regularize=False
):
    """
    Create a single hidden layer autoencoder with relu after encoder.
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :param L:
    :param l1_regularize: set to True if want to regularize the output of the encoder
    :return:
    """
    # Input
    input_signal = Input(
        shape=(input_dim[0], input_dim[1], 1), name="input"
    )  # Input placeholder

    # Build encoder
    if l1_regularize:
        encoder_layer = Conv2D(
            filters=num_conv,
            kernel_size=dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            activity_regularizer=regularizers.l1(0.00001),
            input_shape=(input_dim[0], input_dim[1], 1),
            name="encoder",
        )

    else:
        encoder_layer = Conv2D(
            filters=num_conv,
            kernel_size=dictionary_dim,
            padding="valid",
            use_bias=False,
            activation="relu",
            trainable=True,
            input_shape=(input_dim, 1),
            name="encoder",
        )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding2D(
        padding=(
            (dictionary_dim[0] - 1, dictionary_dim[0] - 1),
            (dictionary_dim[1] - 1, dictionary_dim[1] - 1),
        ),
        name="zeropad",
    )(encoded)

    # Build decoder
    decoder_layer = Conv2D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        name="decoder",
    )
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Divided by L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_constrained_linear_autoencoder_1d(input_dim, dictionary_dim, num_conv, L):
    """
    Create single layer, purely linear autoencoder
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :param L:
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(shape=(input_dim, 1), name="input")  # Input placeholder

    # Build encoder
    encoder_layer = Conv1D(
        filters=num_conv,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        input_shape=(input_dim, 1),
        name="encoder",
    )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding1D(padding=(dictionary_dim - 1), name="zeropad")(encoded)

    # Build decoder
    decoder_layer = Conv1DFlip(encoder_layer, name="decoder")
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Divided by L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_constrained_linear_autoencoder_2d(input_dim, dictionary_dim, num_conv):
    """
    Create single layer, purely linear autoencoder
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(
        shape=(input_dim[0], input_dim[1], 1), name="input"
    )  # Input placeholder

    # Build encoder
    encoder_layer = Conv2D(
        filters=num_conv,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        input_shape=(input_dim[0], input_dim[1], 1),
        name="encoder",
    )
    encoded = encoder_layer(input_signal)

    # Zero-pad
    encoded_pad = ZeroPadding2D(
        padding=(
            (dictionary_dim[0] - 1, dictionary_dim[0] - 1),
            (dictionary_dim[1] - 1, dictionary_dim[1] - 1),
        ),
        name="zeropad",
    )(encoded)

    # Build decoder
    decoder_layer = Conv2DFlip(encoder_layer, name="decoder")
    decoded = decoder_layer(encoded_pad)  # No activation applied

    # Divided by L
    L_layer = Lambda(lambda x: x / L, name="L")
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, encoded)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder


def create_constrained_nonlinear_autoencoder_1d(
    input_dim, dictionary_dim, num_conv, L, lambda_trainable, twosided
):
    """
    Create single layer, non-linear autoencoder
    :param input_dim:
    :param dictionary_dim:
    :param num_conv:
    :return: (autoencoder, encoder)
    """
    # Input
    input_signal = Input(shape=(input_dim, 1), name="input")  # Input placeholder

    # Build encoder
    encoder_layer = Conv1D(
        filters=num_conv,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=True,
        input_shape=(input_dim, 1),
        name="encoder",
    )
    encoded = encoder_layer(input_signal)

    # Build L_layer
    L_layer = Lambda(lambda x: x / L, name="L")

    # Apply trainable ReLu layer
    ttrelu = TrainableThresholdRelu(
        (input_dim - dictionary_dim + 1, num_conv),
        num_conv,
        L,
        lambda_trainable,
        twosided,
        name="ttrelu",
    )(encoded)

    # Zero-pad
    encoded_pad = ZeroPadding1D(padding=(dictionary_dim - 1), name="zeropad")(ttrelu)
    # Build decoder
    decoder_layer = Conv1DFlip(encoder_layer, name="decoder")

    decoded = decoder_layer(encoded_pad)

    # Divided by L
    decoded = L_layer(decoded)

    # encoder only model
    encoder = Model(input_signal, ttrelu)

    # create autoencoder model
    autoencoder = Model(input_signal, decoded)

    return autoencoder, encoder
