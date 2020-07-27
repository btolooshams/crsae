"""
Copyright (c) 2019 CRISP

classes related to TLAE, etc.

:author: Bahareh Tolooshams
"""

import numpy as np
import time
from time import gmtime, strftime
import h5py
import copy
import random
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger
from keras.layers import Conv1D, Conv2D, Input, ZeroPadding2D
from keras.layers import Dense, Lambda, ZeroPadding1D, Add, Subtract
from keras.models import Model
from keras.constraints import max_norm

import sys

sys.path.append("..")

from src.layers.trainable_threshold_relu_layers import TrainableThresholdRelu
from src.callbacks.clr_callback import CyclicLR
from src.callbacks.lrfinder_callback import LRFinder

PATH = "../"


class adam_optimizer:
    def __init__(
        self,
        lr=0.0001,
        amsgrad=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        lambda_lr=0.5,
    ):
        self.name = "adam"
        self.lr = lr
        self.amsgrad = amsgrad
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.lambda_lr = lambda_lr

        self.update_optimizer()

    def update_optimizer(self):
        self.keras_optimizer = Adam(
            lr=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            decay=self.decay,
            amsgrad=self.amsgrad,
        )
        self.keras_optimizer_for_lambda = Adam(
            lr=self.lambda_lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            decay=self.decay,
            amsgrad=self.amsgrad,
        )

    def set_lr(self, lr):
        self.lr = lr
        self.update_optimizer()

    def set_lambda_lr(self, lambda_lr):
        self.lambda_lr = lambda_lr
        self.update_optimizer()

    def set_beta_1(self, beta_1):
        self.beta_1 = beta_1
        self.update_optimizer()

    def set_beta_2(self, beta_2):
        self.beta_2 = beta_2
        self.update_optimizer()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.update_optimizer()

    def set_decay(self, decay):
        self.decay = decay
        self.update_optimizer()

    def set_amsgrad(self, amsgrad):
        self.amsgrad = amsgrad
        self.update_optimizer()

    def get_name(self):
        return self.name

    def get_lr(self):
        return self.lr

    def get_lambda_lr(self):
        return self.lambda_lr

    def get_beta_1(self):
        return self.beta_1

    def get_beta_2(self):
        return self.beta_2

    def get_epsilon(self):
        return self.epsilon

    def get_decay(self):
        return self.decay

    def get_amsgrad(self):
        return self.amsgrad

    def get_keras_optimizer(self):
        return self.keras_optimizer

    def get_keras_optimizer_for_lambda(self):
        return self.keras_optimizer_for_lambda


class trainer:
    def __init__(
        self,
        lambda_trainable,
        num_epochs=10,
        num_val_shuffle=1,
        batch_size=32,
        verbose=1,
    ):
        # training parameters
        self.lambda_trainable = lambda_trainable
        self.num_epochs = num_epochs
        self.num_val_shuffle = num_val_shuffle
        self.batch_size = batch_size
        self.verbose = verbose

        self.val_split = 0.9
        self.unique_number = int(time.time())
        self.fit_time = 0
        self.loss = "mse"
        self.history = []
        self.H_epochs = []
        self.lambda_epochs = []
        self.noiseSTD_epochs = []
        self.close = False
        self.augment = False

        self.reset_callbacks()

        # default optimizer is Adam
        self.optimizer = adam_optimizer()

    def add_best_val_loss_callback(self, loss_type):
        self.loss_type = loss_type
        self.callbacks.append(
            ModelCheckpoint(
                filepath="weights_{}.hdf5".format(self.unique_number),
                monitor=self.loss_type,
                verbose=self.verbose,
                save_best_only=True,
                save_weights_only=True,
            )
        )

    def add_all_epochs_callback(self, loss_type):
        self.loss_type = loss_type
        self.callbacks.append(
            ModelCheckpoint(
                filepath="weights-improvement-%i-{epoch:01d}.hdf5" % self.unique_number,
                monitor=self.loss_type,
                verbose=0,
                save_weights_only=True,
            )
        )

    def add_earlystopping_callback(self, min_delta, patience, loss_type):
        self.earlystopping = True
        self.min_delta = min_delta
        self.patience = patience
        self.loss_type = loss_type
        self.callbacks.append(
            EarlyStopping(
                monitor=self.loss_type,
                min_delta=self.min_delta,
                patience=self.patience,
                verbose=0,
                mode="auto",
            )
        )

    def add_cyclic_lr_callback(self, base_lr, max_lr, step_size):
        self.cycleLR = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.callbacks.append(
            CyclicLR(
                base_lr=self.base_lr,
                max_lr=self.max_lr,
                step_size=self.step_size,
                mode="triangular",
            )
        )

    def add_progressbar_callback(self):
        self.callbacks.append(ProgbarLogger())

    def reset_callbacks(self):
        self.earlystopping = False
        self.cycleLR = False
        self.callbacks = []

    def get_callbacks(self):
        return self.callbacks

    def get_verbose(self):
        return self.verbose

    def get_batch_size(self):
        return self.batch_size

    def get_num_epochs(self):
        return self.num_epochs

    def get_loss(self):
        return self.loss

    def get_callbacks(self):
        return self.callbacks

    def get_callback_loss_type(self):
        return self.loss_type

    def get_fit_time(self):
        return self.fit_time

    def get_val_split(self):
        return self.val_split

    def get_unique_number(self):
        return self.unique_number

    def get_history(self):
        return self.history

    def get_H_epochs(self):
        return self.H_epochs

    def get_lambda_epochs(self):
        return self.lambda_epochs

    def get_noiseSTD_epochs(self):
        return self.noiseSTD_epochs

    def get_close(self):
        return self.close

    def get_augment(self):
        return self.augment

    # optimizer
    def set_optimizer(self, name):
        if name == "adam":
            self.optimizer = adam_optimizer()

    def get_optimizer(self):
        return self.optimizer

    def set_loss(self, loss):
        self.loss = loss

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def set_fit_time(self, fit_time):
        self.fit_time = fit_time

    def set_val_split(self, val_split):
        self.val_split = val_split

    def set_history(self, history):
        if self.lambda_trainable:
            val_loss = history["val_loss"]
            val_loss = [float(i) for i in val_loss]

            val_lambda_loss = history["val_lambda_loss"]
            val_lambda_loss = [float(i) for i in val_lambda_loss]

            train_loss = history["train_loss"]
            train_loss = [float(i) for i in train_loss]

            train_lambda_loss = history["train_lambda_loss"]
            train_lambda_loss = [float(i) for i in train_lambda_loss]
        else:
            val_loss = history.history["val_loss"]
            val_loss = [float(i) for i in val_loss]

            train_loss = history.history["loss"]
            train_loss = [float(i) for i in train_loss]

        if self.lambda_trainable:
            history = {
                "val_loss": val_loss,
                "val_lambda_loss": val_lambda_loss,
                "train_loss": train_loss,
                "train_lambda_loss": train_lambda_loss,
            }
        else:
            history = {"val_loss": val_loss, "train_loss": train_loss}
        self.history = history

    def set_close(self, close):
        self.close = close

    def set_augment(self, augment):
        self.augment = augment


class TLAE_1d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        twosided,
        lambda_trainable,
        alpha,
        num_channels,
        lambda_uncertainty=1,
    ):
        """
        Initializes CRsAE 1d with model and training parameters.

        """
        # models parameters
        self.input_dim = input_dim
        self.num_conv = num_conv
        self.dictionary_dim = dictionary_dim
        self.num_iterations = num_iterations
        self.twosided = twosided
        self.lambda_trainable = lambda_trainable
        self.Ne = self.input_dim - self.dictionary_dim + 1
        self.alpha = alpha
        self.num_channels = num_channels
        self.data_space = 1
        self.lambda_uncertainty = lambda_uncertainty

        # initialize trainer
        self.trainer = trainer(self.lambda_trainable)

    def build_model(self, noiseSTD):
        print("build model.")
        # compute lambda from noise level
        self.noiseSTD = np.zeros((1,)) + noiseSTD
        lambda_donoho = np.zeros(
            (self.num_conv,), dtype=np.float32
        ) + noiseSTD * np.sqrt(2 * np.log(self.num_conv * self.Ne))
        self.lambda_donoho = np.copy(lambda_donoho)

        # build model
        build_graph_start_time = time.time()
        residual, encoder = self.TLAE_1d_model()
        build_graph_time = time.time() - build_graph_start_time
        print("build_graph_time:", np.round(build_graph_time, 4), "s")
        # this is for training purposes with alpha
        residual_for_train, temp = self.TLAE_1d_model()

        residual.get_layer("soft_threshold").set_weights([self.lambda_donoho])
        residual_for_train.get_layer("soft_threshold").set_weights(
            [self.lambda_donoho * self.alpha]
        )

        self.encoder = encoder
        self.residual = residual

        self.residual_for_train = residual_for_train

        # initialize H
        self.initialize_H()

    def TLAE_1d_model(self):
        """
        Create DAG for transform learning auto-encoder (TLAE).
        :return: (residual, encoder)
        """
        y = Input(shape=(self.input_dim, 1), name="y")

        H = Conv1D(
            filters=self.num_conv,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            input_shape=(self.input_dim, 1),
            name="H",
            kernel_constraint=max_norm(max_value=1, axis=0),
        )
        # Apply trainable ReLu layer
        soft_threshold_layer = TrainableThresholdRelu(
            (self.input_dim - self.dictionary_dim + 1, self.num_conv),
            self.num_conv,
            1,
            self.lambda_trainable,
            self.twosided,
            name="soft_threshold",
        )

        # Hy
        Hy = H(y)

        z = soft_threshold_layer(Hy)

        # residual
        res = Subtract(name="residual")([Hy, z])

        encoder = Model(y, z)
        residual = Model(y, res)

        return residual, encoder

    def initialize_H(self):
        self.H = np.random.randn(self.dictionary_dim, 1, self.num_conv)
        self.H /= np.linalg.norm(self.H, axis=0)
        # set H for autoencoder and encoder
        self.residual.get_layer("H").set_weights([self.H])
        # set H for model2 if lambda is lambda_trainable
        if not self.lambda_trainable:
            self.residual_for_train.get_layer("H").set_weights([self.H])

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
        self.H /= self.H
        # set HT in autoencoder
        self.residual.get_layer("H").set_weights([self.H])
        if not self.lambda_trainable:
            self.residual_for_train.get_layer("H").set_weights([self.H])

    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value
        self.residual.get_layer("soft_threshold").set_weights([self.lambda_value])
        self.residual_for_train.get_layer("soft_threshold").set_weights(
            [self.lambda_value]
        )

    def get_H(self):
        return self.residual.get_layer("H").get_weights()[0]

    def get_lambda(self):
        lambda_value = self.residual.get_layer("soft_threshold").get_weights()[0]
        return lambda_value

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
        # load parameters from the best val_loss
        self.residual_for_train.load_weights(
            "weights_{}.hdf5".format(self.trainer.get_unique_number())
        )
        self.set_H(self.residual_for_train.get_layer("H").get_weights()[0])

    def update_H_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.H_epochs = []
        for epoch in range(num_epochs):
            self.residual.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.H_epochs.append(self.residual.get_layer("H").get_weights()[0])

    def encode(self, y):
        return self.encoder.predict(y)

    def compile(self):
        def logdet_loss_function(y_true, y_pred):
            return K.sum(y_pred, axis=-1)

        loss = self.trainer.get_loss()

        self.residual_for_train.compile(
            optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
        )

    def fit(self, partial_y_train, y_val, lr_finder=[], num_epochs=4):
        train_zero_vector = np.zeros((partial_y_train.shape[0], self.Ne, self.num_conv))
        val_zero_vector = np.zeros((y_val.shape[0], self.Ne, self.num_conv))
        if lr_finder:
            history = self.residual_for_train.fit(
                partial_y_train,
                train_zero_vector,
                epochs=num_epochs,
                batch_size=self.trainer.get_batch_size(),
                validation_data=(y_val, val_zero_vector),
                verbose=self.trainer.get_verbose(),
                shuffle=True,
                callbacks=[lr_finder],
            )
        else:
            history = self.residual_for_train.fit(
                partial_y_train,
                train_zero_vector,
                epochs=self.trainer.get_num_epochs(),
                batch_size=self.trainer.get_batch_size(),
                validation_data=(y_val, val_zero_vector),
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
            "../experiments/{}/results/TLAE_results_lr.h5".format(folder_name), "w"
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
        # set the trained weights in autoencoder
        self.update_H_after_training()

    def train_and_save(self, y_train, folder_name):
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
        # fit (train)
        history = self.fit(partial_y_train, y_val)
        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_H_epochs()
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
            "{}/experiments/{}/results/TLAE_results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])

        hf.create_dataset("H_epochs", data=H_epochs)
        # hf.create_dataset("lambda_epochs", data=lambda_epochs)
        # hf.create_dataset("noiseSTD_epochs", data=noiseSTD_epochs)
        hf.create_dataset("H_learned", data=H_learned)
        hf.create_dataset("lambda_learned", data=lambda_value)
        hf.create_dataset("lambda_donoho", data=self.lambda_donoho)
        hf.close()

        return time
