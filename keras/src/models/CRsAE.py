"""
Copyright (c) 2019 CRISP

classes related to CRsAE, etc.

:author: Bahareh Tolooshams
"""

import numpy as np
import time
from time import gmtime, strftime
import h5py
import copy
import random
import keras.backend as K
from keras.layers import Conv1D, Conv2D, Input, ZeroPadding2D
from keras.layers import Dense, Lambda, ZeroPadding1D, Flatten
from keras.models import Model
from keras.constraints import max_norm

import sys

sys.path.append("..")

from src.trainers.trainers import trainer
from src.layers.conv_tied_layers import Conv1DFlip, Conv2DFlip
from src.layers.ista_fista_layers import FISTA_1d, FISTA_2d
from src.layers.trainable_lambda_loss_function_layers import TrainableLambdaLossFunction
from src.callbacks.clr_callback import CyclicLR
from src.callbacks.lrfinder_callback import LRFinder

PATH = "../"


class CRsAE_1d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        L,
        twosided,
        lambda_trainable,
        alpha,
        num_channels,
        noiseSTD_trainable=False,
        lambda_EM=False,
        delta=100,
        lambda_single=False,
        noiseSTD_lr=0.01,
    ):
        """
        Initializes CRsAE 1d with model and training parameters.

        """
        # models parameters
        self.input_dim = input_dim
        self.num_conv = num_conv
        self.dictionary_dim = dictionary_dim
        self.num_iterations = num_iterations
        self.L = L
        self.twosided = twosided
        self.lambda_trainable = lambda_trainable
        self.Ne = self.input_dim - self.dictionary_dim + 1
        self.alpha = alpha
        self.num_channels = num_channels
        self.noiseSTD_trainable = noiseSTD_trainable
        self.data_space = 1
        self.delta = delta
        self.lambda_single = lambda_single
        self.lambda_EM = lambda_EM
        self.noiseSTD_lr = noiseSTD_lr

        # initialize trainer
        self.trainer = trainer(self.lambda_trainable)

    def build_model(self, noiseSTD):
        print("build model.")
        # compute lambda from noise level
        self.noiseSTD = np.zeros((1,)) + noiseSTD
        lambda_donoho = np.sqrt(2 * np.log(self.num_conv * self.Ne)) / noiseSTD
        self.lambda_donoho = np.copy(lambda_donoho)

        if self.lambda_trainable:
            # build model
            build_graph_start_time = time.time()
            autoencoder, model_2, encoder, decoder = self.CRsAE_1d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            if self.lambda_EM:
                if self.lambda_single:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [np.zeros((1,), dtype=np.float32) + self.lambda_donoho]
                    )
                else:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        ]
                    )
            else:
                if self.lambda_single:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
                else:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
        else:
            # build model
            build_graph_start_time = time.time()
            autoencoder, encoder, decoder = self.CRsAE_1d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            # this is for training purposes with alpha
            autoencoder_for_train, temp, temp = self.CRsAE_1d_model()
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (np.zeros((1,), dtype=np.float32) + self.lambda_donoho)
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        )
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.decoder = decoder

        if self.lambda_trainable:
            self.model_2 = model_2
        else:
            self.autoencoder_for_train = autoencoder_for_train

        # initialize H
        self.initialize_H()

    def CRsAE_1d_model(self):
        """
        Create DAG for constraint reccurent sparse auto-encoder (CRsAE).
        :return: (autoencoder, encoder, decoder)
        """
        y = Input(shape=(self.input_dim, 1), name="y")

        # Zero-pad layer
        padding_layer = ZeroPadding1D(padding=(self.dictionary_dim - 1), name="zeropad")

        if self.lambda_trainable:
            ###########
            # model 1
            ###########
            # Dictionary for model 1 (trainable)
            HT_1 = Conv1D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim, 1),
                name="HT",
                # kernel_constraint=max_norm(max_value=1, axis=0),
            )
            # HTy for model 1
            HTy_1 = HT_1(y)
            # initialize z0 to be 0 vector for model 1
            z0_model1 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_1)
            # Have to define the transpose layer after creating a flowgraph with HT for model 1
            H_model1 = Conv1DFlip(HT_1, name="H")
            # perform FISTA for num_iterations
            # FISTA layer for the model trianing H
            FISTA_layer_model_1 = FISTA_1d(
                HT_1,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            # z_T output of soft-thresholding
            zt_model1 = FISTA_layer_model_1(z0_model1)

            # zeropad zt for model 1
            zt_padded_1 = padding_layer(zt_model1)
            # reconstruct y for model 1
            y_hat_model1 = H_model1(zt_padded_1)

            # build encoder for model 1
            encoder_model1 = Model(y, zt_model1)
            # build autoencoder for model 1
            autoencoder_model1 = Model(y, y_hat_model1)

            ###########
            # model 2
            ###########
            if self.lambda_EM:
                z = Input(shape=(self.num_conv,), name="l1_norm_code")
                logLoss_layer = TrainableLambdaLossFunction(
                    self.Ne,
                    self.num_conv,
                    self.lambda_donoho,
                    self.delta,
                    self.lambda_single,
                    name="logLambdaLoss",
                )
                log_loss = logLoss_layer(z)
                model_2 = Model(z, log_loss)
            else:
                HT_2 = Conv1D(
                    filters=self.num_conv,
                    kernel_size=self.dictionary_dim,
                    padding="valid",
                    use_bias=False,
                    activation=None,
                    trainable=False,
                    input_shape=(self.input_dim, 1),
                    name="HT",
                    # kernel_constraint=max_norm(max_value=1, axis=0),
                )
                # HTy for model 2
                HTy_2 = HT_2(y)
                # initialize z0 to be 0 vector for model 2
                z0_model2 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_2)
                # Have to define the transpose layer after creating a flowgraph with HT for model 2
                H_model2 = Conv1DFlip(HT_2, name="H")
                # FISTA layer for the model trianing lambda
                FISTA_layer_model_2 = FISTA_1d(
                    HT_2,
                    y,
                    self.L,
                    True,
                    self.twosided,
                    self.num_iterations,
                    self.lambda_single,
                    self.lambda_EM,
                    name="FISTA",
                )
                # z_T output of soft-thresholding
                zt_model2, lambda_term_model2 = FISTA_layer_model_2(z0_model2)
                # for model 2
                lambda_placeholder = Lambda(
                    lambda x: x[:, 0, :] * 0 + lambda_term_model2, name="lambda"
                )(z0_model2)
                lambda_zt = Lambda(lambda x: x * lambda_term_model2, name="l1_norm")(
                    zt_model2
                )

                # build model for model 2
                # zeropad zt for model 2
                zt_padded_2 = padding_layer(zt_model2)
                # reconstruct y for model 2
                y_hat_model2 = H_model2(zt_padded_2)
                model_2 = Model(y, [lambda_zt, lambda_placeholder, lambda_placeholder])

            # for decoding
            input_code = Input(shape=(self.Ne, self.num_conv), name="input_code")
            input_code_padded = padding_layer(input_code)
            decoded = H_model1(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder_model1, model_2, encoder_model1, decoder
        else:
            HT = Conv1D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim, 1),
                name="HT",
                kernel_constraint=max_norm(max_value=1, axis=0),
            )
            # HTy
            HTy = HT(y)
            # initialize z0 to be 0 vector
            z0 = Lambda(lambda x: x * 0, name="initialize_z")(HTy)

            # Have to define the transpose layer after creating a flowgraph with HT
            H = Conv1DFlip(HT, name="H")
            # perform FISTA for num_iterations
            # FISTA layer
            FISTA_layer = FISTA_1d(
                HT,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            zt = FISTA_layer(z0)
            # zeropad zt
            zt_padded = padding_layer(zt)
            # reconstruct y
            y_hat = H(zt_padded)

            # build encoder and autoencoder
            encoder = Model(y, zt)
            autoencoder = Model(y, y_hat)

            # for decoding
            input_code = Input(
                shape=(self.input_dim - self.dictionary_dim + 1, self.num_conv),
                name="input_code",
            )
            input_code_padded = padding_layer(input_code)
            decoded = H(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder, encoder, decoder

    def initialize_H(self):
        self.H = np.random.randn(self.dictionary_dim, 1, self.num_conv)
        self.H /= np.linalg.norm(self.H, axis=0)
        # set H for autoencoder and encoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        # set H for model2 if lambda is lambda_trainable
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_H(self, H, H_noisestd=0):
        if np.sum(H_noisestd):
            H_noisy = np.copy(H)
            for n in range(self.num_conv):
                H_noisy[:, :, n] += H_noisestd[n] * np.random.randn(
                    self.dictionary_dim, 1
                )
                self.H = H_noisy
        else:
            self.H = H
        self.H /= np.linalg.norm(self.H, axis=0)
        # set HT in autoencoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_lambda(self, lambda_value):
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )
        if self.lambda_trainable:
            if self.lambda_EM:
                self.model_2.get_layer("logLambdaLoss").set_weights([lambda_value])
            else:
                noiseSTD_estimate = self.model_2.get_layer("FISTA").get_weights()[1]
                self.model_2.get_layer("FISTA").set_weights(
                    [lambda_value, noiseSTD_estimate]
                )
        else:
            noiseSTD_estimate = self.autoencoder_for_train.get_layer(
                "FISTA"
            ).get_weights()[1]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD_estimate]
            )

    def set_noiseSTD(self, noiseSTD):
        lambda_value = self.autoencoder.get_layer("FISTA").get_weights()[0]
        self.autoencoder.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        if self.lambda_trainable:
            if not self.lambda_EM:
                lambda_value = self.model_2.get_layer("FISTA").get_weights()[0]
                self.model_2.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        else:
            lambda_value = self.autoencoder_for_train.get_layer("FISTA").get_weights()[
                0
            ]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD]
            )

    def get_H(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_H_estimate(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_lambda(self):
        return self.autoencoder.get_layer("FISTA").get_weights()[0]

    def get_lambda_estimate(self):
        if self.lambda_EM:
            return self.model_2.get_layer("logLambdaLoss").get_weights()[0]
        else:
            return self.model_2.get_layer("FISTA").get_weights()[0]

    def get_noiseSTD(self):
        return self.autoencoder.get_layer("FISTA").get_weights()[1]

    def get_input_dim(self):
        return self.input_dim

    def get_num_conv(self):
        return self.num_conv

    def get_dictionary_dim(self):
        return self.dictionary_dim

    def get_num_iterations(self):
        return self.num_iterations

    def get_L(self):
        return self.L

    def get_twosided(self):
        return self.twosided

    def get_alpha(self):
        return self.alpha

    def get_num_channels(self):
        return self.num_channels

    def get_data_space(self):
        return self.data_space

    def update_H_after_training(self):
        if self.lambda_trainable:
            # load parameters from the best val_loss
            self.autoencoder.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder.get_layer("HT").get_weights()[0])
        else:
            # load parameters from the best val_loss
            self.autoencoder_for_train.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder_for_train.get_layer("HT").get_weights()[0])

    def update_H_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.H_epochs = []
        for epoch in range(num_epochs):
            if self.lambda_trainable:
                self.autoencoder.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder.get_layer("HT").get_weights()[0]
                )
            else:
                self.autoencoder_for_train.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder_for_train.get_layer("HT").get_weights()[0]
                )

    def update_H(self):
        H_estimate = self.get_H_estimate()
        self.H = H_estimate
        self.model_2.get_layer("HT").set_weights([self.H])

    def update_lambda(self):
        lambda_value = self.get_lambda_estimate()
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )

    def update_lambda_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.lambda_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.lambda_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[0]
            )

    def update_noiseSTD_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.noiseSTD_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.noiseSTD_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[1][0]
            )

    def encode(self, y):
        return self.encoder.predict(y)

    def decode(self, z):
        return self.decoder.predict(z)

    def separate(self, y):
        # get z
        z = self.encode(y)

        y_hat_separate = np.zeros((y.shape[0], self.input_dim, self.num_conv))
        for n in range(self.num_conv):
            temp = np.copy(np.zeros((z.shape[0], self.Ne, self.num_conv)))
            temp[:, :, n] = np.copy(z[:, :, n])
            decoded = copy.deepcopy(self.decode(temp))
            y_hat_separate[:, :, n] = np.squeeze(np.copy(decoded), axis=2)
        return y_hat_separate

    def denoise(self, y):
        return self.autoencoder.predict(y)

    def compile(self):
        def lambda_loss_function(y_true, y_pred):
            return K.sum(y_pred, axis=-1)

        def log_lambda_loss_function(y_true, y_pred):
            return K.sum(K.log(y_pred), axis=-1)

        def lambda_gamma_prior_loss_function(y_true, y_pred):
            delta = self.delta
            r = delta * self.lambda_donoho
            return K.sum((r - 1) * K.log(y_pred) - delta * y_pred, axis=-1)

        loss = self.trainer.get_loss()

        if self.lambda_trainable:
            self.autoencoder.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(),
                loss=loss,
                loss_weights=[1 / 2],
            )
            if self.lambda_EM:
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=lambda_loss_function,
                )
            else:
                loss_model2 = [
                    "mae",
                    log_lambda_loss_function,
                    lambda_gamma_prior_loss_function,
                ]
                l1_norm_lambda_loss_weight = 1
                log_lambda_loss_weight = -self.Ne
                lambda_gamma_prior_loss_weight = -1
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=loss_model2,
                    loss_weights=[
                        l1_norm_lambda_loss_weight,
                        log_lambda_loss_weight,
                        lambda_gamma_prior_loss_weight,
                    ],
                )

        else:
            self.autoencoder_for_train.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
            )

    def train_on_batch(self, training_batch_i):
        # single gradient update for H
        x, y, sample_weights = self.autoencoder._standardize_user_data(
            training_batch_i, training_batch_i, sample_weight=None, class_weight=None
        )
        if self.autoencoder._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_1 = self.autoencoder.train_function(ins)

        if self.lambda_EM:
            training_batch_i_code = self.encode(training_batch_i)
            l1_norm_training_batch_i_code = np.mean(
                np.sum(np.abs(training_batch_i_code), axis=1), axis=0
            )
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                np.zeros((1, self.num_conv)) + l1_norm_training_batch_i_code,
                [np.zeros((1, 1))],
                sample_weight=None,
                class_weight=None,
            )
        else:
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                training_batch_i,
                [
                    np.zeros((training_batch_i.shape[0], self.Ne, self.num_conv)),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                ],
                sample_weight=None,
                class_weight=None,
            )

        if self.model_2._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_2 = self.model_2.train_function(ins)

    def fit(self, partial_y_train, y_val, lr_finder=[], num_epochs=4):
        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("noiseSTD:", self.get_noiseSTD()[0])
                print("lambda:", self.get_lambda())
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :
                    ]

                    if self.noiseSTD_trainable:
                        # get noiseSTD (l+1) estimate from H (l), lambda (l), noiseSTD (l) by gradient descent
                        # get reconstruction loss on batch
                        training_batch_i_hat = self.denoise(training_batch_i)

                        noiseSTD_estimate_past = self.get_noiseSTD()[0]
                        beta = 1 - self.noiseSTD_lr
                        noise_estimate_from_batch = np.sqrt(
                            np.mean(
                                np.mean(
                                    np.square(
                                        np.squeeze(
                                            training_batch_i - training_batch_i_hat
                                        )
                                    ),
                                    axis=1,
                                )
                            )
                        )
                        noiseSTD_estimate = np.zeros((1,)) + (
                            beta * noiseSTD_estimate_past
                            + (1 - beta) * noise_estimate_from_batch
                        )

                    # train on single batch
                    self.train_on_batch(training_batch_i)
                    #
                    # if self.lambda_EM:
                    # # this is for updating lambda
                    # beta = 1 - self.lambda_lr
                    # x_training_batch_i = self.encode(training_batch_i)
                    # # update lambda by solving the minimization problem
                    # l1_norm_training_batch_i_hat = np.mean(np.sum(np.sum(np.abs(x_training_batch_i),axis=2),axis=1),axis=0)
                    #
                    # lambda_past = self.get_lambda()
                    # lambda_estimate_from_batch = np.zeros((self.num_conv,), dtype=np.float32) + (self.Ne * self.num_conv / (l1_norm_training_batch_i_hat+1e-7))
                    #
                    # lambda_new = beta * lambda_past + (1-beta) * lambda_estimate_from_batch
                    # self.set_lambda(lambda_new)
                    # # this is for test of updating lambda
                    # beta = 0.94  # for 32 samples in batch
                    # x_training_batch_i = self.encode(training_batch_i)
                    # # update lambda by solving the minimization problem
                    # l1_norm_training_batch_i_hat = np.mean(
                    #     np.sum(np.abs(x_training_batch_i), axis=1),
                    #     axis=0,
                    # )
                    # lambda_past = self.get_lambda()
                    # lambda_estimate_from_batch = self.Ne / l1_norm_training_batch_i_hat
                    # lambda_new = (
                    #     beta * lambda_past + (1 - beta) * lambda_estimate_from_batch
                    # )
                    # self.set_lambda(lambda_new)

                    # # this is for test of updating lambda
                    # x_training_batch_i = self.encode(training_batch_i)
                    # # update lambda by solving the minimization problem
                    # l1_norm_training_batch_i_hat = np.mean(np.sum(np.abs(x_training_batch_i),axis=1),axis=0)
                    # # print('l1 norm:',l1_norm_training_batch_i_hat)
                    # lambda_past = self.get_lambda()
                    # lambda_gradient = l1_norm_training_batch_i_hat - (self.Ne / (lambda_past+1e-7))
                    # print('gradient:',lambda_gradient)
                    # lambda_new = lambda_past - 5e-1 * lambda_gradient
                    # self.set_lambda(lambda_new)

                    # # update lambda on autoencoder
                    self.update_lambda()
                    # # # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()
                    else:
                        print("ERROR: EM implementation is not complete!")

                    # normalize filters
                    temp = self.get_H()
                    self.set_H(temp)

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)

                # keep track of weights for all epochs
                self.autoencoder.save_weights(
                    "weights-improvement-%i-%i.hdf5"
                    % (self.trainer.get_unique_number(), epoch + 1),
                    overwrite=True,
                )

                # test on validation set
                y_val_hat = self.denoise(y_val)
                val_loss_i = np.mean(
                    np.mean(np.square(np.squeeze(y_val - y_val_hat)), axis=1), axis=0
                )
                # get training error
                partial_y_train_hat = self.denoise(partial_y_train)
                train_loss_i = np.mean(
                    np.mean(
                        np.square(np.squeeze(partial_y_train - partial_y_train_hat)),
                        axis=1,
                    ),
                    axis=0,
                )

                # lambda loss on validation set
                partial_x_train_hat = self.encode(partial_y_train)
                l1_norm_partial_x_train_hat = np.mean(
                    np.sum(np.abs(partial_x_train_hat), axis=1), axis=0
                )
                if self.lambda_EM:
                    train_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_partial_x_train_hat
                    )
                else:
                    train_lambda_loss_i = 0

                # lambda loss on training set
                x_val_hat = self.encode(y_val)
                l1_norm_x_val_hat = np.mean(np.sum(np.abs(x_val_hat), axis=1), axis=0)
                if self.lambda_EM:
                    val_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_x_val_hat
                    )
                else:
                    val_lambda_loss_i = 0

                val_loss.append(val_loss_i)
                val_lambda_loss.append(train_lambda_loss_i)
                train_loss.append(train_loss_i)
                train_lambda_loss.append(train_lambda_loss_i)

                print(
                    "Epoch %d/%d" % (epoch + 1, self.trainer.get_num_epochs()),
                    "loss:",
                    np.round(train_loss_i, 8),
                    "lambda_loss:",
                    np.round(train_lambda_loss_i, 3),
                    "val_loss:",
                    np.round(val_loss_i, 8),
                    "val_lambda_loss:",
                    np.round(val_lambda_loss_i, 3),
                )

                if epoch == 0:
                    self.autoencoder.save_weights(
                        "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        overwrite=True,
                    )
                    print(
                        "val_loss improved from inf to %.8f saving model to %s"
                        % (
                            val_loss_i,
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        )
                    )
                else:
                    if val_loss_i < min_val_loss:
                        self.autoencoder.save_weights(
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                            overwrite=True,
                        )
                        print(
                            "val_loss improved from %.8f to %.8f saving model to %s"
                            % (
                                val_loss[-2],
                                val_loss_i,
                                "weights_{}.hdf5".format(
                                    self.trainer.get_unique_number()
                                ),
                            )
                        )
                    else:
                        print("val_loss NOT improved from %.8f" % (min_val_loss))

                min_val_loss = min(val_loss)

            history = {
                "val_loss": val_loss,
                "val_lambda_loss": val_lambda_loss,
                "train_loss": train_loss,
                "train_lambda_loss": train_lambda_loss,
            }
        else:
            if lr_finder:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=num_epochs,
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=[lr_finder],
                )
            else:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=self.trainer.get_num_epochs(),
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=self.trainer.get_callbacks(),
                )
        return history

    def find_lr(self, y_train, folder_name, num_epochs=4, min_lr=1e-7, max_lr=1e-1):
        # This implementation is ONLY for LR for autoencoder (H training) not lambda model (model 2)
        print("find lr.")

        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]
        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # lr callback
        epoch_size = int(0.9 * num_train)
        lr_finder = LRFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_per_epoch=np.ceil(epoch_size / self.trainer.get_batch_size()),
            epochs=num_epochs,
        )

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder, num_epochs)

        # save lr results
        hf = h5py.File(
            "../experiments/{}/results/results_lr.h5".format(folder_name), "w"
        )
        hf.create_dataset("iterations", data=lr_finder.get_iterations())
        hf.create_dataset("lr", data=lr_finder.get_lr())
        hf.create_dataset("loss_lr", data=lr_finder.get_loss())
        hf.close()

    def augment_data(self, y):
        # flip the data
        y_flip = -1 * y
        # circular shift the data
        y_cirshift = np.roll(y, np.random.randint(1, y.shape[1] - 1), axis=1)
        return np.concatenate([y, y_flip, y_cirshift], axis=0)

    def check_speed(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # compile model with loss (weighted loss)
        self.compile()

        print("start training.")
        fit_start_time = time.time()

        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("epoch %s/%s" % (epoch + 1, self.trainer.get_num_epochs()))
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :
                    ]

                    # train on single batch
                    self.train_on_batch(training_batch_i)

                    # update lambda on autoencoder
                    self.update_lambda()
                    # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)
        else:
            history = self.autoencoder_for_train.fit(
                partial_y_train,
                partial_y_train,
                epochs=self.trainer.get_num_epochs(),
                batch_size=self.trainer.get_batch_size(),
                validation_data=(y_val, y_val),
                verbose=0,
                shuffle=True,
            )

        fit_time = time.time() - fit_start_time
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        return fit_time

    def train(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        print("start training.")
        fit_start_time = time.time()

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder, num_epochs)

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
        if self.lambda_trainable:
            # set all lambda and nosieSTD epochs
            self.update_lambda_epochs()
            self.update_noiseSTD_epochs()
        # set the trained weights in autoencoder
        self.update_H_after_training()

    def train_and_save(self, y_train, folder_name):
        num_train = y_train.shape[0]

        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)

        # indices = np.load("../experiments/{}/data/indices.npy".format(folder_name))

        y_val = y_train[indices[partial_train_num:], :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # compile model with loss (weighted loss)
        self.compile()

        print("start training.")
        fit_start_time = time.time()
        # fit (train)
        history = self.fit(partial_y_train, y_val)
        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
        if self.lambda_trainable:
            # set all lambda and nosieSTD epochs
            self.update_lambda_epochs()
            self.update_noiseSTD_epochs()
        # set the trained weights in autoencoder
        self.update_H_after_training()
        # save results
        folder_time = self.save_results(folder_name)

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        return folder_time

    def save_results(self, folder_name, time=1.234):
        print("save results.")
        # get history results
        history = self.trainer.get_history()
        # get H epochs
        H_epochs = self.trainer.get_H_epochs()
        if self.lambda_trainable:
            # get lambda epochs
            lambda_epochs = self.trainer.get_lambda_epochs()
            # get noiseSTD epochs
            noiseSTD_epochs = self.trainer.get_noiseSTD_epochs()
        # get H result
        H_learned = self.get_H()
        # get lambda
        lambda_value = self.get_lambda()
        # write in h5 file
        if time == 1.234:
            time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hf = h5py.File(
            "{}/experiments/{}/results/results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )

        if self.trainer.get_cycleLR():
            hf.create_dataset("lr_iterations", data=self.trainer.get_lr_iterations())
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])
        if self.lambda_trainable:
            hf.create_dataset("val_lambda_loss", data=history["val_lambda_loss"])
            hf.create_dataset("train_lambda_loss", data=history["train_lambda_loss"])
            hf.create_dataset("lambda_epochs", data=lambda_epochs)
            hf.create_dataset("noiseSTD_epochs", data=noiseSTD_epochs)

        hf.create_dataset("H_epochs", data=H_epochs)
        hf.create_dataset("H_learned", data=H_learned)
        hf.create_dataset("lambda_learned", data=lambda_value)
        hf.create_dataset("lambda_donoho", data=self.lambda_donoho)
        hf.close()

        return time


class CRsAE_2d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        L,
        twosided,
        lambda_trainable,
        alpha,
        num_channels,
        noiseSTD_trainable=False,
        lambda_EM=False,
        delta=100,
        lambda_single=False,
        noiseSTD_lr=0.01,
    ):
        """
        Initializes CRsAE 2d with model and training parameters.

        """
        # models parameters
        self.input_dim = input_dim
        self.num_conv = num_conv
        self.dictionary_dim = dictionary_dim
        self.num_iterations = num_iterations
        self.L = L
        self.twosided = twosided
        self.lambda_trainable = lambda_trainable
        self.Ne = (self.input_dim[0] - self.dictionary_dim[0] + 1) * (
            self.input_dim[1] - self.dictionary_dim[1] + 1
        )
        self.alpha = alpha
        self.num_channels = num_channels
        self.noiseSTD_trainable = noiseSTD_trainable
        self.data_space = 2
        self.lambda_single = lambda_single
        self.lambda_EM = lambda_EM
        self.noiseSTD_lr = noiseSTD_lr

        # initialize trainer
        self.trainer = trainer(self.lambda_trainable)

    def build_model(self, noiseSTD):
        print("build model.")

        self.noiseSTD = np.zeros((1,)) + noiseSTD
        # compute lambda from noise level
        lambda_donoho = np.sqrt(2 * np.log(self.num_conv * self.Ne)) / noiseSTD
        self.lambda_donoho = np.copy(lambda_donoho)

        if self.lambda_trainable:
            # build model
            build_graph_start_time = time.time()
            autoencoder, encoder, decoder = self.CRsAE_2d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
            if self.lambda_EM:
                if self.lambda_single:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [np.zeros((1,), dtype=np.float32) + self.lambda_donoho]
                    )
                else:
                    model_2.get_layer("logLambdaLoss").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        ]
                    )
            else:
                if self.lambda_single:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
                else:
                    model_2.get_layer("FISTA").set_weights(
                        [
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho,
                            self.noiseSTD,
                        ]
                    )
        else:
            # build model
            build_graph_start_time = time.time()
            autoencoder, encoder, decoder = self.CRsAE_2d_model()
            build_graph_time = time.time() - build_graph_start_time
            print("build_graph_time:", np.round(build_graph_time, 4), "s")
            # this is for training purposes with alpha
            autoencoder_for_train, temp, temp = self.CRsAE_2d_model()
            if self.lambda_single:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((1,), dtype=np.float32) + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (np.zeros((1,), dtype=np.float32) + self.lambda_donoho)
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )
            else:
                autoencoder.get_layer("FISTA").set_weights(
                    [
                        np.zeros((self.num_conv,), dtype=np.float32)
                        + self.lambda_donoho,
                        self.noiseSTD,
                    ]
                )
                autoencoder_for_train.get_layer("FISTA").set_weights(
                    [
                        (
                            np.zeros((self.num_conv,), dtype=np.float32)
                            + self.lambda_donoho
                        )
                        * self.alpha,
                        self.noiseSTD,
                    ]
                )

        self.classifier = self.CRsAE_2d_classification_model()

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.decoder = decoder
        self.autoencoder_for_train = autoencoder_for_train

        if self.lambda_trainable:
            self.model_2 = model_2
        else:
            self.autoencoder_for_train = autoencoder_for_train

        # initialize H
        self.initialize_H()

    def CRsAE_2d_model(self):
        """
        Create DAG for constraint reccurent sparse auto-encoder (CRsAE).
        :return: (autoencoder, encoder, decoder)
        """
        y = Input(shape=(self.input_dim[0], self.input_dim[1], 1), name="y")

        # Zero-pad layer
        padding_layer = ZeroPadding2D(
            padding=(
                (self.dictionary_dim[0] - 1, self.dictionary_dim[0] - 1),
                (self.dictionary_dim[1] - 1, self.dictionary_dim[1] - 1),
            ),
            name="zeropad",
        )

        if self.lambda_trainable:
            ###########
            # model 1
            ###########
            # Dictionary for model 1 (trainable)
            HT = Conv2D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim[0], self.input_dim[1], 1),
                name="HT",
                kernel_constraint=max_norm(max_value=1),
            )
            # HTy
            HTy_1 = HT_1(y)
            # initialize z0 to be 0 vector
            z0_model1 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_1)
            # Have to define the transpose layer after creating a flowgraph with HT
            H_model1 = Conv2DFlip(HT_1, name="H")
            # perform FISTA for num_iterations
            # FISTA layer for the model trianing H
            FISTA_layer_model_1 = FISTA_2d(
                HT_1,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            # z_T output of soft-thresholding
            zt_model1 = FISTA_layer_model_1(z0_model1)

            # zeropad zt for model 1
            zt_padded_1 = padding_layer(zt_model1)
            # reconstruct y for model 1
            y_hat_model1 = H_model1(zt_padded_1)

            # build encoder for model 1
            encoder_model1 = Model(y, zt_model1)
            # build autoencoder for model 1
            autoencoder_model1 = Model(y, y_hat_model1)

            ###########
            # model 2
            ###########
            if self.lambda_EM:
                z = Input(shape=(self.num_conv,), name="l1_norm_code")
                logLoss_layer = TrainableLambdaLossFunction(
                    self.Ne,
                    self.num_conv,
                    self.lambda_donoho,
                    self.delta,
                    self.lambda_single,
                    name="logLambdaLoss",
                )
                log_loss = logLoss_layer(z)
                model_2 = Model(z, log_loss)
            else:
                HT_2 = Conv2D(
                    filters=self.num_conv,
                    kernel_size=self.dictionary_dim,
                    padding="valid",
                    use_bias=False,
                    activation=None,
                    trainable=False,
                    input_shape=(self.input_dim[0], self.input_dim[1], 1),
                    name="HT",
                    # kernel_constraint=max_norm(max_value=1, axis=0),
                )
                # HTy for model 2
                HTy_2 = HT_2(y)
                # initialize z0 to be 0 vector for model 2
                z0_model2 = Lambda(lambda x: x * 0, name="initialize_z")(HTy_2)
                # Have to define the transpose layer after creating a flowgraph with HT for model 2
                H_model2 = Conv2DFlip(HT_2, name="H")
                # FISTA layer for the model trianing lambda
                FISTA_layer_model_2 = FISTA_2d(
                    HT_2,
                    y,
                    self.L,
                    True,
                    self.twosided,
                    self.num_iterations,
                    self.lambda_single,
                    self.lambda_EM,
                    name="FISTA",
                )
                # z_T output of soft-thresholding
                zt_model2, lambda_term_model2 = FISTA_layer_model_2(z0_model2)
                # for model 2
                lambda_placeholder = Lambda(
                    lambda x: x[:, 0, 0, :] * 0 + lambda_term_model2, name="lambda"
                )(z0_model2)
                lambda_zt = Lambda(lambda x: x * lambda_term_model2, name="l1_norm")(
                    zt_model2
                )

                # build model for model 2
                # zeropad zt for model 2
                zt_padded_2 = padding_layer(zt_model2)
                # reconstruct y for model 2
                y_hat_model2 = H_model2(zt_padded_2)
                model_2 = Model(y, [lambda_zt, lambda_placeholder, lambda_placeholder])

            # for decoding
            input_code = Input(
                shape=(
                    self.input_dim[0] - self.dictionary_dim[0] + 1,
                    self.input_dim[1] - self.dictionary_dim[1] + 1,
                    self.num_conv,
                ),
                name="input_code",
            )
            input_code_padded = padding_layer(input_code)
            decoded = H_model1(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder_model1, model_2, encoder_model1, decoder
        else:
            HT = Conv2D(
                filters=self.num_conv,
                kernel_size=self.dictionary_dim,
                padding="valid",
                use_bias=False,
                activation=None,
                trainable=True,
                input_shape=(self.input_dim[0], self.input_dim[1], 1),
                name="HT",
                kernel_constraint=max_norm(max_value=1, axis=0),
            )
            # HTy
            HTy = HT(y)
            # initialize z0 to be 0 vector
            z0 = Lambda(lambda x: x * 0, name="initialize_z")(HTy)

            # Have to define the transpose layer after creating a flowgraph with HT
            H = Conv2DFlip(HT, name="H")
            # perform FISTA for num_iterations
            # FISTA layer
            FISTA_layer = FISTA_2d(
                HT,
                y,
                self.L,
                False,
                self.twosided,
                self.num_iterations,
                self.lambda_single,
                False,
                name="FISTA",
            )
            zt = FISTA_layer(z0)
            # zeropad zt
            zt_padded = padding_layer(zt)
            # reconstruct y
            y_hat = H(zt_padded)

            # build encoder and autoencoder
            encoder = Model(y, zt)
            autoencoder = Model(y, y_hat)

            # for decoding
            input_code = Input(
                shape=(
                    self.input_dim[0] - self.dictionary_dim[0] + 1,
                    self.input_dim[1] - self.dictionary_dim[1] + 1,
                    self.num_conv,
                ),
                name="input_code",
            )
            input_code_padded = padding_layer(input_code)
            decoded = H(input_code_padded)

            decoder = Model(input_code, decoded)

            return autoencoder, encoder, decoder

    def CRsAE_2d_classification_model(self):
        """
        Create DAG for constraint reccurent sparse auto-encoder (CRsAE).
        :return: (classifier)
        """
        y = Input(shape=(self.input_dim[0], self.input_dim[1], 1), name="y")

        # Zero-pad layer
        padding_layer = ZeroPadding2D(
            padding=(
                (self.dictionary_dim[0] - 1, self.dictionary_dim[0] - 1),
                (self.dictionary_dim[1] - 1, self.dictionary_dim[1] - 1),
            ),
            name="zeropad",
        )

        HT = Conv2D(
            filters=self.num_conv,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=False,
            input_shape=(self.input_dim[0], self.input_dim[1], 1),
            name="HT",
            kernel_constraint=max_norm(max_value=1, axis=0),
        )
        # HTy
        HTy = HT(y)
        # initialize z0 to be 0 vector
        z0 = Lambda(lambda x: x * 0, name="initialize_z")(HTy)

        # Have to define the transpose layer after creating a flowgraph with HT
        H = Conv2DFlip(HT, name="H")
        # perform FISTA for num_iterations
        # FISTA layer
        FISTA_layer = FISTA_2d(
            HT,
            y,
            self.L,
            False,
            self.twosided,
            self.num_iterations,
            self.lambda_single,
            False,
            name="FISTA",
        )
        zt = FISTA_layer(z0)

        ###########
        # classification model
        ###########
        flat_code = Flatten()(zt)
        flat_code_normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(flat_code)
        # model = Dense(units=64,activation='relu', name='D1')(flat_code)
        label = Dense(units=10, activation="softmax", name="D1")(flat_code_normalized)

        classifier = Model(y, label)

        return classifier

    def initialize_H(self):
        self.H = np.random.randn(
            self.dictionary_dim[0], self.dictionary_dim[1], 1, self.num_conv
        )
        self.H /= np.linalg.norm(self.H, "fro", axis=(0, 1))
        # set H for autoencoder and encoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        # set H for model2 if lambda is lambda_trainable
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])

    def set_H(self, H, H_noisestd=0):
        if np.sum(H_noisestd):
            H_noisy = np.copy(H)
            for n in range(self.num_conv):
                H_noisy[:, :, :, n] += H_noisestd[n] * np.random.randn(
                    self.dictionary_dim[0], self.dictionary_dim[1], 1
                )
                self.H = H_noisy
        else:
            self.H = H
        self.H /= np.linalg.norm(self.H, "fro", axis=(0, 1))
        # set HT in autoencoder
        self.autoencoder.get_layer("HT").set_weights([self.H])
        if self.lambda_trainable:
            if not self.lambda_EM:
                self.model_2.get_layer("HT").set_weights([self.H])
        else:
            self.autoencoder_for_train.get_layer("HT").set_weights([self.H])
        self.classifier.get_layer("HT").set_weights([self.H])

    def set_lambda(self, lambda_value):
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )
        if self.lambda_trainable:
            if self.lambda_EM:
                self.model_2.get_layer("logLambdaLoss").set_weights([lambda_value])
            else:
                noiseSTD_estimate = self.model_2.get_layer("FISTA").get_weights()[1]
                self.model_2.get_layer("FISTA").set_weights(
                    [lambda_value, noiseSTD_estimate]
                )
        else:
            noiseSTD_estimate = self.autoencoder_for_train.get_layer(
                "FISTA"
            ).get_weights()[1]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD_estimate]
            )

    def set_noiseSTD(self, noiseSTD):
        lambda_value = self.autoencoder.get_layer("FISTA").get_weights()[0]
        self.autoencoder.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        if self.lambda_trainable:
            if not self.lambda_EM:
                lambda_value = self.model_2.get_layer("FISTA").get_weights()[0]
                self.model_2.get_layer("FISTA").set_weights([lambda_value, noiseSTD])
        else:
            lambda_value = self.autoencoder_for_train.get_layer("FISTA").get_weights()[
                0
            ]
            self.autoencoder_for_train.get_layer("FISTA").set_weights(
                [lambda_value, noiseSTD]
            )

    def get_H(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_H_estimate(self):
        return self.autoencoder.get_layer("HT").get_weights()[0]

    def get_lambda(self):
        return self.autoencoder.get_layer("FISTA").get_weights()[0]

    def get_lambda_estimate(self):
        if self.lambda_EM:
            return self.model_2.get_layer("logLambdaLoss").get_weights()[0]
        else:
            return self.model_2.get_layer("FISTA").get_weights()[0]

    def get_noiseSTD(self):
        return self.autoencoder.get_layer("FISTA").get_weights()[1]

    def get_input_dim(self):
        return self.input_dim

    def get_num_conv(self):
        return self.num_conv

    def get_dictionary_dim(self):
        return self.dictionary_dim

    def get_num_iterations(self):
        return self.num_iterations

    def get_L(self):
        return self.L

    def get_twosided(self):
        return self.twosided

    def get_alpha(self):
        return self.alpha

    def get_num_channels(self):
        return self.num_channels

    def get_data_space(self):
        return self.data_space

    def update_H_after_training(self):
        if self.lambda_trainable:
            # load parameters from the best val_loss
            self.autoencoder.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder.get_layer("HT").get_weights()[0])
        else:
            # load parameters from the best val_loss
            self.autoencoder_for_train.load_weights(
                "weights_{}.hdf5".format(self.trainer.get_unique_number())
            )
            self.set_H(self.autoencoder_for_train.get_layer("HT").get_weights()[0])

    def update_H_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.H_epochs = []
        for epoch in range(num_epochs):
            if self.lambda_trainable:
                self.autoencoder.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder.get_layer("HT").get_weights()[0]
                )
            else:
                self.autoencoder_for_train.load_weights(
                    "weights-improvement-{}-{}.hdf5".format(
                        self.trainer.get_unique_number(), epoch + 1
                    )
                )
                self.trainer.H_epochs.append(
                    self.autoencoder_for_train.get_layer("HT").get_weights()[0]
                )

    def update_H(self):
        H_estimate = self.get_H_estimate()
        self.H = H_estimate
        self.model_2.get_layer("HT").set_weights([self.H])

    def update_lambda(self):
        lambda_value = self.get_lambda_estimate()
        noiseSTD_estimate = self.autoencoder.get_layer("FISTA").get_weights()[1]
        self.autoencoder.get_layer("FISTA").set_weights(
            [lambda_value, noiseSTD_estimate]
        )

    def update_lambda_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.lambda_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.lambda_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[0]
            )

    def update_noiseSTD_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.noiseSTD_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.noiseSTD_epochs.append(
                self.autoencoder.get_layer("FISTA").get_weights()[1][0]
            )

    def encode(self, y):
        return self.encoder.predict(y)

    def decode(self, z):
        return self.decoder.predict(z)

    def separate(self, y):
        # get z
        z = self.encode(y)

        y_hat_separate = np.zeros(
            (y.shape[0], self.input_dim[0], self.input_dim[1], self.num_conv)
        )
        for n in range(self.num_conv):
            temp = np.copy(
                np.zeros(
                    (
                        z.shape[0],
                        self.input_dim[0] - self.dictionary_dim[0] + 1,
                        self.input_dim[1] - self.dictionary_dim[1] + 1,
                        self.num_conv,
                    )
                )
            )
            temp[:, :, :, n] = np.copy(z[:, :, :, n])
            decoded = copy.deepcopy(self.decode(temp))
            y_hat_separate[:, :, :, n] = np.squeeze(np.copy(decoded), axis=3)
        return y_hat_separate

    def denoise(self, y):
        return self.autoencoder.predict(y)

    def compile(self):
        def log_lambda_loss_function(y_true, y_pred):
            return K.sum(K.log(y_pred), axis=-1)

        def lambda_prior_loss_function(y_true, y_pred):
            return K.sum(K.square(y_pred - y_true), axis=-1)

        def lambda_gamma_prior_loss_function(y_true, y_pred):
            delta = self.delta
            r = delta * self.lambda_donoho
            return K.sum((r - 1) * K.log(y_pred) - delta * y_pred, axis=-1)

        loss = self.trainer.get_loss()

        if self.lambda_trainable:
            self.autoencoder.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(),
                loss=loss,
                loss_weights=[1 / 2],
            )
            if self.lambda_EM:
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=lambda_loss_function,
                )
            else:
                loss_model2 = [
                    "mae",
                    log_lambda_loss_function,
                    lambda_gamma_prior_loss_function,
                ]
                l1_norm_lambda_loss_weight = 1
                log_lambda_loss_weight = -self.Ne
                lambda_gamma_prior_loss_weight = -1
                self.model_2.compile(
                    optimizer=self.trainer.optimizer.get_keras_optimizer_for_lambda(),
                    loss=loss_model2,
                    loss_weights=[
                        l1_norm_lambda_loss_weight,
                        log_lambda_loss_weight,
                        lambda_gamma_prior_loss_weight,
                    ],
                )

        else:
            self.autoencoder_for_train.compile(
                optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
            )

    def train_on_batch(self, training_batch_i):
        # single gradient update for H
        x, y, sample_weights = self.autoencoder._standardize_user_data(
            training_batch_i, training_batch_i, sample_weight=None, class_weight=None
        )
        if self.autoencoder._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_1 = self.autoencoder.train_function(ins)

        if self.lambda_EM:
            training_batch_i_code = self.encode(training_batch_i)
            l1_norm_training_batch_i_code = np.mean(
                np.sum(np.sum(np.abs(training_batch_i_code), axis=1), axis=1), axis=0
            )
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                np.zeros((1, self.num_conv)) + l1_norm_training_batch_i_code,
                [np.zeros((1, 1))],
                sample_weight=None,
                class_weight=None,
            )
        else:
            # single gradient update for lambda
            x, y, sample_weights = self.model_2._standardize_user_data(
                training_batch_i,
                [
                    np.zeros(
                        (
                            training_batch_i.shape[0],
                            self.input_dim[0] - self.dictionary_dim[0] + 1,
                            self.input_dim[1] - self.dictionary_dim[1] + 1,
                            self.num_conv,
                        )
                    ),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                    np.zeros((training_batch_i.shape[0], self.num_conv)),
                ],
                sample_weight=None,
                class_weight=None,
            )
        if self.model_2._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [1.0]
        else:
            ins = x + y + sample_weights
        output_2 = self.model_2.train_function(ins)

    def fit(self, partial_y_train, y_val, lr_finder=[], num_epochs=4):
        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_l1_norm_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("noiseSTD:", self.get_noiseSTD()[0])
                print("lambda:", self.get_lambda())
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :, :
                    ]

                    if self.noiseSTD_trainable:
                        # get reconstruction loss on batch
                        training_batch_i_hat = self.denoise(training_batch_i)

                        noiseSTD_estimate_past = self.get_noiseSTD()[0]
                        beta = 1 - self.noiseSTD_lr
                        noise_estimate_from_batch = np.sqrt(
                            np.mean(
                                np.mean(
                                    np.mean(
                                        np.square(
                                            np.squeeze(
                                                training_batch_i - training_batch_i_hat
                                            )
                                        ),
                                        axis=1,
                                    ),
                                    axis=1,
                                )
                            )
                        )
                        noiseSTD_estimate = np.zeros((1,)) + (
                            beta * noiseSTD_estimate_past
                            + (1 - beta) * noise_estimate_from_batch
                        )

                    # train on single batch
                    self.train_on_batch(training_batch_i)
                    # update lambda on autoencoder
                    self.update_lambda()
                    # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()
                    else:
                        print("ERROR: EM implementation is not complete!")

                    # normalize filters
                    temp = self.get_H()
                    self.set_H(temp)

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)

                # keep track of weighs for all epochs
                self.autoencoder.save_weights(
                    "weights-improvement-%i-%i.hdf5"
                    % (self.trainer.get_unique_number(), epoch + 1),
                    overwrite=True,
                )

                # test on validation set
                y_val_hat = self.denoise(y_val)
                val_loss_i = np.mean(
                    np.mean(np.square(np.squeeze(y_val - y_val_hat)), axis=1), axis=0
                )
                # get training error
                partial_y_train_hat = self.denoise(partial_y_train)
                train_loss_i = np.mean(
                    np.mean(
                        np.mean(
                            np.square(
                                np.squeeze(partial_y_train - partial_y_train_hat)
                            ),
                            axis=1,
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                # lambda loss on validation set
                partial_x_train_hat = self.encode(partial_y_train)
                l1_norm_partial_x_train_hat = np.mean(
                    np.sum(np.sum(np.abs(partial_x_train_hat), axis=1), axis=1), axis=0
                )
                if self.lambda_EM:
                    train_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_partial_x_train_hat
                    )
                else:
                    train_lambda_loss_i = 0

                # lambda loss on training set
                x_val_hat = self.encode(y_val)
                l1_norm_x_val_hat = np.mean(
                    np.sum(np.sum(np.abs(x_val_hat), axis=1), axis=1), axis=0
                )
                if self.lambda_EM:
                    val_lambda_loss_i = self.model_2.predict(
                        np.zeros((1, self.num_conv)) + l1_norm_x_val_hat
                    )
                else:
                    val_lambda_loss_i = 0

                val_loss.append(val_loss_i)
                val_lambda_loss.append(train_lambda_loss_i)
                train_loss.append(train_loss_i)
                train_lambda_loss.append(train_lambda_loss_i)

                print(
                    "Epoch %d/%d" % (epoch + 1, self.trainer.get_num_epochs()),
                    "loss:",
                    np.round(train_loss_i, 8),
                    "lambda_loss:",
                    np.round(train_lambda_loss_i, 3),
                    "val_loss:",
                    np.round(val_loss_i, 8),
                    "val_lambda_loss:",
                    np.round(val_lambda_loss_i, 3),
                )

                if epoch == 0:
                    self.autoencoder.save_weights(
                        "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        overwrite=True,
                    )
                    print(
                        "val_loss improved from inf to %.8f saving model to %s"
                        % (
                            val_loss_i,
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                        )
                    )
                else:
                    if val_loss_i < min_val_loss:
                        self.autoencoder.save_weights(
                            "weights_{}.hdf5".format(self.trainer.get_unique_number()),
                            overwrite=True,
                        )
                        print(
                            "val_loss improved from %.8f to %.8f saving model to %s"
                            % (
                                val_loss[-2],
                                val_loss_i,
                                "weights_{}.hdf5".format(
                                    self.trainer.get_unique_number()
                                ),
                            )
                        )
                    else:
                        print("val_loss NOT improved from %.8f" % (min_val_loss))

                min_val_loss = min(val_loss)

            history = {
                "val_loss": val_loss,
                "val_lambda_loss": val_lambda_loss,
                "train_loss": train_loss,
                "train_lambda_loss": train_lambda_loss,
            }
        else:
            if lr_finder:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=num_epochs,
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=[lr_finder],
                )
            else:
                history = self.autoencoder_for_train.fit(
                    partial_y_train,
                    partial_y_train,
                    epochs=self.trainer.get_num_epochs(),
                    batch_size=self.trainer.get_batch_size(),
                    validation_data=(y_val, y_val),
                    verbose=self.trainer.get_verbose(),
                    shuffle=True,
                    callbacks=self.trainer.get_callbacks(),
                )
        return history

    def find_lr(self, y_train, folder_name, num_epochs=4, min_lr=1e-7, max_lr=1e-1):
        print("find lr.")

        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :, :]
        partial_y_train_original = y_train[indices[:partial_train_num], :, :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # lr callback
        epoch_size = int(0.9 * num_train)
        lr_finder = LRFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_per_epoch=np.ceil(epoch_size / self.trainer.get_batch_size()),
            epochs=num_epochs,
        )
        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val, lr_finder)

        # save lr results
        hf = h5py.File(
            "../experiments/{}/results/results_lr.h5".format(folder_name), "w"
        )
        hf.create_dataset("iterations", data=lr_finder.get_iterations())
        hf.create_dataset("lr", data=lr_finder.get_lr())
        hf.create_dataset("loss_lr", data=lr_finder.get_loss())
        hf.close()

    def augment_data(self, y):
        # flip the data
        y_flip = -1 * y
        # circular shift the data
        # y_cirshift = np.roll(y,np.random.randint(1,y.shape[1]-1),axis=1)
        print(np.random.randint(1, y.shape[1]))
        # return np.concatenate([y,y_flip,y_cirshift],axis=0)
        return np.concatenate([y, y_flip], axis=0)

    def check_speed(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)
        y_val = y_train[indices[partial_train_num:], :, :, :]

        partial_y_train_original = y_train[indices[:partial_train_num], :, :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        # compile model with loss (weighted loss)
        self.compile()

        print("start training.")
        fit_start_time = time.time()

        if self.lambda_trainable:
            num_batches = np.ceil(
                partial_y_train.shape[0] / self.trainer.get_batch_size()
            )
            val_loss = []
            train_loss = []
            val_lambda_loss = []
            train_lambda_loss = []
            print(
                "Train on %i samples, validate on %i samples"
                % (partial_y_train.shape[0], y_val.shape[0])
            )

            # this is something to do for compile (before start training)
            self.autoencoder._make_train_function()
            self.model_2._make_train_function()

            for epoch in range(self.trainer.get_num_epochs()):
                print("epoch %s/%s" % (epoch + 1, self.trainer.get_num_epochs()))
                batches = np.linspace(0, num_batches - 1, num_batches)
                random.shuffle(batches)
                for batch in batches:
                    batch_begin_index = np.int(batch * self.trainer.get_batch_size())
                    batch_end_index = np.int(
                        (batch + 1) * self.trainer.get_batch_size()
                    )
                    training_batch_i = partial_y_train[
                        batch_begin_index:batch_end_index, :, :, :
                    ]

                    # train on single batch
                    self.train_on_batch(training_batch_i)

                    # update lambda on autoencoder
                    self.update_lambda()
                    # update H on model 2
                    if not self.lambda_EM:
                        self.update_H()

                    # update noiseSTD
                    if self.noiseSTD_trainable:
                        self.set_noiseSTD(noiseSTD_estimate)
        else:
            history = self.autoencoder_for_train.fit(
                partial_y_train,
                partial_y_train,
                epochs=self.trainer.get_num_epochs(),
                batch_size=self.trainer.get_batch_size(),
                validation_data=(y_val, y_val),
                verbose=0,
                shuffle=True,
            )

        fit_time = time.time() - fit_start_time
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        return fit_time

    def train(self, y_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)

        y_val = y_train[indices[partial_train_num:], :, :, :]
        partial_y_train_original = y_train[indices[:partial_train_num], :, :, :]
        if self.trainer.get_augment():
            partial_y_train = self.augment_data(partial_y_train_original)
        else:
            partial_y_train = partial_y_train_original

        print("start training.")
        fit_start_time = time.time()

        # compile model with loss (weighted loss)
        self.compile()
        # fit (train)
        history = self.fit(partial_y_train, y_val)

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")
        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
        if self.lambda_trainable:
            # set all lambda and nosieSTD epochs
            self.update_lambda_epochs()
            self.update_noiseSTD_epochs()
        # set the trained weights in autoencoder
        self.update_H_after_training()

    def train_classification(self, y_train, label_train):
        num_train = y_train.shape[0]
        # divide train data into train and val sets
        partial_train_num = int(self.trainer.get_val_split() * num_train)
        # shuffle the training data
        indices = np.arange(0, num_train, 1)
        np.random.shuffle(indices)

        print("start training.")
        fit_start_time = time.time()

        self.classifier.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # fit (train)
        history = self.classifier.fit(
            y_train, label_train, epochs=5, batch_size=256, validation_split=0.1
        )

    def evaluate_classifier(self, y_test, label_test):
        return self.classifier.evaluate(y_test, label_test)

    def save_results(self, folder_name, time=1.234):
        print("save results.")
        # get history results
        history = self.trainer.get_history()
        # get H epochs
        H_epochs = self.trainer.get_H_epochs()
        if self.lambda_trainable:
            # get lambda epochs
            lambda_epochs = self.trainer.get_lambda_epochs()
            # get noiseSTD epochs
            noiseSTD_epochs = self.trainer.get_noiseSTD_epochs()
        # get H result
        H_learned = self.get_H()
        # get lambda
        lambda_value = self.get_lambda()
        # write in h5 file
        if time == 1.234:
            time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hf = h5py.File(
            "{}/experiments/{}/results/results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )

        if self.trainer.get_cycleLR():
            hf.create_dataset("lr_iterations", data=self.trainer.get_lr_iterations())
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])
        if self.lambda_trainable:
            hf.create_dataset("val_lambda_loss", data=history["val_lambda_loss"])
            hf.create_dataset("train_lambda_loss", data=history["train_lambda_loss"])
            hf.create_dataset("lambda_epochs", data=lambda_epochs)
            hf.create_dataset("noiseSTD_epochs", data=noiseSTD_epochs)
            hf.create_dataset("lambda_learned", data=lambda_value)

        # hf.create_dataset("H_epochs", data=H_epochs)
        hf.create_dataset("H_learned", data=H_learned)
        hf.create_dataset("lambda_donoho", data=self.lambda_donoho)
        hf.close()

        return time
