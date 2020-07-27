"""
Copyright (c) 2019 CRISP

classes related to LCSC, etc.

:author: Bahareh Tolooshams
"""

import numpy as np
import time
from time import gmtime, strftime
import h5py
import copy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger
from keras.layers import Conv1D, Conv2D, Input, ZeroPadding2D
from keras.layers import Dense, Lambda, ZeroPadding1D, Add, Subtract
from keras.models import Model
from keras.constraints import max_norm

import sys

sys.path.append("..")

from src.layers.conv_tied_layers import Conv1DFlip, Conv2DFlip
from src.layers.trainable_threshold_relu_layers import (
    TrainableThresholdRelu,
    TrainableThresholdRelu_learned,
)
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
    ):
        self.name = "adam"
        self.lr = lr
        self.amsgrad = amsgrad
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay

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

    def set_lr(self, lr):
        self.lr = lr
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


class trainer:
    def __init__(self, num_epochs=10, num_val_shuffle=1, batch_size=32, verbose=1):
        # training parameters
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

    def get_fit_time(self):
        return self.fit_time

    def get_val_split(self):
        return self.val_split

    def get_unique_number(self):
        return self.unique_number

    def get_history(self):
        return self.history

    def get_Wd_epochs(self):
        return self.Wd_epochs

    def get_We_epochs(self):
        return self.We_epochs

    def get_d_epochs(self):
        return self.d_epochs

    def get_lambda_epochs(self):
        return self.lambda_epochs

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
        val_loss = history.history["val_loss"]
        val_loss = [float(i) for i in val_loss]

        train_loss = history.history["loss"]
        train_loss = [float(i) for i in train_loss]

        history = {"val_loss": val_loss, "train_loss": train_loss}
        self.history = history

    def set_close(self, close):
        self.close = close

    def set_augment(self, augment):
        self.augment = augment


class LCSC_1d:
    def __init__(
        self,
        input_dim,
        num_conv,
        dictionary_dim,
        num_iterations,
        L,
        twosided,
        lambda_trainable,
    ):
        """
        Initializes LCSC 1d with model and training parameters.

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
        self.data_space = 1

        # initialize trainer
        self.trainer = trainer()

    def build_model(self):
        print("build model.")

        # build model
        build_graph_start_time = time.time()
        autoencoder, encoder, decoder = self.LCSC_1d_model()
        build_graph_time = time.time() - build_graph_start_time
        print("build_graph_time:", np.round(build_graph_time, 4), "s")

        autoencoder, encoder, decoder = self.LCSC_1d_model()

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.decoder = decoder

        # initialize weights
        self.initialize_weights()

    def LCSC_1d_model(self):
        """
        Create DAG for learned convolutional sparse coding (LCSC).
        :return: (autoencoder, encoder, decoder)
        """
        y = Input(shape=(self.input_dim, 1), name="y")
        We = Conv1D(
            filters=self.num_conv,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            input_shape=(self.input_dim, 1),
            name="We",
        )
        # Wey
        Wey = We(y)

        # Apply trainable ReLu layer
        if self.lambda_trainable:
            soft_threshold_layer = TrainableThresholdRelu_learned(
                (self.input_dim - self.dictionary_dim + 1, self.num_conv),
                self.num_conv,
                self.twosided,
                name="soft_threshold",
            )
        else:
            soft_threshold_layer = TrainableThresholdRelu(
                (self.input_dim - self.dictionary_dim + 1, self.num_conv),
                self.num_conv,
                self.L,
                self.lambda_trainable,
                self.twosided,
                name="soft_threshold",
            )

        # initialize z0
        z0 = soft_threshold_layer(Wey)
        zt = [z0]

        # Zero-pad layer
        padding_layer = ZeroPadding1D(padding=(self.dictionary_dim - 1), name="zeropad")

        # Wd layer
        Wd = Conv1D(
            filters=1,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            name="Wd",
        )
        # Lista Iterations
        for t in range(1, self.num_iterations):
            zt.append(
                self.learned_convolutional_LISTA_iteration_1d(
                    zt[t - 1], y, Wd, We, soft_threshold_layer, t, padding_layer
                )
            )

        # Build a separate convolution matrix d for linear decoder for reconstruction
        d = Conv1D(
            filters=1,
            kernel_size=self.dictionary_dim,
            padding="valid",
            use_bias=False,
            activation=None,
            trainable=True,
            name="d",
        )

        z_padded = padding_layer(zt[self.num_iterations - 1])
        # reconstruct y
        y_hat = d(z_padded)

        encoder = Model(y, zt[self.num_iterations - 1])
        autoencoder = Model(y, y_hat)

        # for decoding
        input_code = Input(
            shape=(self.input_dim - self.dictionary_dim + 1, self.num_conv),
            name="input_code",
        )
        input_code_padded = padding_layer(input_code)
        decoded = d(input_code_padded)

        decoder = Model(input_code, decoded)

        return autoencoder, encoder, decoder

    def learned_convolutional_LISTA_iteration_1d(
        self, z_old, y, Wd, We, soft_threshold_layer, t, padding_layer
    ):
        """
        Helper function for transpose Lista iteration. Take z_old and makes it into z_new.
        :param z_old: output of previous Lista iteration
        :param y: input to encoder
        :param Wd: Conv1D layer that represents Wd
        :param We: Conv1D layer that represents We
        :param soft_threshold_layer: TrainableThresholdRelu layer
        :return: z_new, output of Lista iteration
        """
        z_old_padded = padding_layer(z_old)
        Wd_z_old = Wd(z_old_padded)
        res = Subtract(name="subtract_{}".format(t))([y, Wd_z_old])
        We_res = We(res)
        pre_z_new = Add(name="add_{}".format(t))([z_old, We_res])
        z_new = soft_threshold_layer(pre_z_new)
        return z_new

    def initialize_weights(self):
        # Wd
        self.Wd = np.random.randn(self.dictionary_dim, self.num_conv, 1)
        self.Wd /= np.linalg.norm(self.Wd, axis=0)
        # We
        self.We = (1 / self.L) * np.expand_dims(
            np.flip(np.squeeze(self.Wd), axis=0), axis=1
        )
        # d
        self.d = np.copy(self.Wd)
        # set weights for autoencoder and encoder
        self.autoencoder.get_layer("Wd").set_weights([self.Wd])
        self.autoencoder.get_layer("We").set_weights([self.We])
        self.autoencoder.get_layer("d").set_weights([self.d])

    def set_weights(self, Wd, We, d):
        self.Wd = Wd
        self.We = We
        self.d = d

        # set weights for autoencoder and encoder
        self.autoencoder.get_layer("Wd").set_weights([self.Wd])
        self.autoencoder.get_layer("We").set_weights([self.We])
        self.autoencoder.get_layer("d").set_weights([self.d])

    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value
        self.autoencoder.get_layer("soft_threshold").set_weights([self.lambda_value])

    def get_Wd(self):
        return self.autoencoder.get_layer("Wd").get_weights()[0]

    def get_We(self):
        return self.autoencoder.get_layer("We").get_weights()[0]

    def get_d(self):
        return self.autoencoder.get_layer("d").get_weights()[0]

    def get_lambda(self):
        return self.autoencoder.get_layer("soft_threshold").get_weights()[0]

    def get_input_dim(self):
        return self.input_dim

    def get_num_conv(self):
        return self.num_conv

    def get_dictionary_dim(self):
        return self.dictionary_dim

    def get_num_iterations(self):
        return self.num_iterations

    def get_twosided(self):
        return self.twosided

    def get_data_space(self):
        return self.data_space

    def update_weights_after_training(self):
        # load parameters from the best val_loss
        self.autoencoder.load_weights(
            "weights_{}.hdf5".format(self.trainer.get_unique_number())
        )
        self.set_weights(
            self.autoencoder.get_layer("Wd").get_weights()[0],
            self.autoencoder.get_layer("We").get_weights()[0],
            self.autoencoder.get_layer("d").get_weights()[0],
        )
        self.set_lambda(self.autoencoder.get_layer("soft_threshold").get_weights()[0])

    def update_weights_epochs(self):
        num_epochs = len(self.trainer.get_history()["val_loss"])
        self.trainer.Wd_epochs = []
        self.trainer.We_epochs = []
        self.trainer.d_epochs = []
        self.trainer.lambda_epochs = []
        for epoch in range(num_epochs):
            self.autoencoder.load_weights(
                "weights-improvement-{}-{}.hdf5".format(
                    self.trainer.get_unique_number(), epoch + 1
                )
            )
            self.trainer.Wd_epochs.append(
                self.autoencoder.get_layer("Wd").get_weights()[0]
            )
            self.trainer.We_epochs.append(
                self.autoencoder.get_layer("We").get_weights()[0]
            )
            self.trainer.d_epochs.append(
                self.autoencoder.get_layer("d").get_weights()[0]
            )
            self.trainer.lambda_epochs.append(
                self.autoencoder.get_layer("soft_threshold").get_weights()[0]
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

    def find_lr(self, y_train, folder_name, num_epochs=4, min_lr=1e-7, max_lr=1e-1):
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
        loss = self.trainer.get_loss()
        # lr callback
        epoch_size = int(0.9 * num_train)
        lr_finder = LRFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_per_epoch=np.ceil(epoch_size / self.trainer.get_batch_size()),
            epochs=num_epochs,
        )
        # train
        self.autoencoder.compile(
            optimizer=self.trainer.optimizer.get_keras_optimizer(),
            loss=self.trainer.get_loss(),
        )
        history = self.autoencoder.fit(
            partial_y_train,
            partial_y_train,
            epochs=num_epochs,
            batch_size=self.trainer.get_batch_size(),
            validation_data=(y_val, y_val),
            verbose=self.trainer.get_verbose(),
            shuffle=True,
            callbacks=[lr_finder],
        )
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
        loss = self.trainer.get_loss()
        print("start training.")
        fit_start_time = time.time()
        self.autoencoder.compile(
            optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
        )

        history = self.autoencoder.fit(
            partial_y_train,
            partial_y_train,
            epochs=self.trainer.get_num_epochs(),
            batch_size=self.trainer.get_batch_size(),
            validation_data=(y_val, y_val),
            verbose=self.trainer.get_verbose(),
            shuffle=True,
            callbacks=self.trainer.get_callbacks(),
        )

        fit_time = time.time() - fit_start_time
        self.trainer.set_fit_time(fit_time)
        print("finish training.")
        print("fit_time:", fit_time / 60, "min")

        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_weights_epochs()
        # set the trained weights in autoencoder
        self.update_weights_after_training()

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
        loss = self.trainer.get_loss()
        self.autoencoder.compile(
            optimizer=self.trainer.optimizer.get_keras_optimizer(), loss=loss
        )

        print("start training.")
        fit_start_time = time.time()
        history = self.autoencoder.fit(
            partial_y_train,
            partial_y_train,
            epochs=self.trainer.get_num_epochs(),
            batch_size=self.trainer.get_batch_size(),
            validation_data=(y_val, y_val),
            verbose=self.trainer.get_verbose(),
            shuffle=True,
            callbacks=self.trainer.get_callbacks(),
        )
        # set hisotry
        self.trainer.set_history(history)
        # set all h epochs
        self.update_weights_epochs()
        # set the trained weights in autoencoder
        self.update_weights_after_training()
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
        # get weights epochs
        Wd_epochs = self.trainer.get_Wd_epochs()
        We_epochs = self.trainer.get_We_epochs()
        d_epochs = self.trainer.get_d_epochs()
        lambda_epochs = self.trainer.get_lambda_epochs()
        # get weights result
        Wd_learned = self.get_Wd()
        We_learned = self.get_We()
        d_learned = self.get_d()
        lambda_learned = self.get_lambda()
        # write in h5 file
        if time == 1.234:
            time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hf = h5py.File(
            "{}/experiments/{}/results/LCSC_results_training_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )
        hf.create_dataset("val_loss", data=history["val_loss"])
        hf.create_dataset("train_loss", data=history["train_loss"])
        hf.create_dataset("Wd_epochs", data=Wd_epochs)
        hf.create_dataset("We_epochs", data=We_epochs)
        hf.create_dataset("d_epochs", data=d_epochs)
        hf.create_dataset("Wd_learned", data=Wd_learned)
        hf.create_dataset("We_learned", data=We_learned)
        hf.create_dataset("d_learned", data=d_learned)
        hf.create_dataset("lambda_epochs", data=lambda_epochs)
        hf.create_dataset("lambda_learned", data=lambda_learned)
        hf.close()

        return time
