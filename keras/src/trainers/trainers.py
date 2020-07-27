"""
Copyright (c) 2019 CRISP

classes related to trainer for CRsAE, etc.

:author: Bahareh Tolooshams
"""

import time
from time import gmtime, strftime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger

import sys

sys.path.append("..")

from src.optimizers.adam_optimizer import adam_optimizer
from src.optimizers.sgd_optimizer import sgd_optimizer
from src.callbacks.clr_callback import CyclicLR

PATH = "../"


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

        self.set_optimizer()

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

    def add_cyclic_lr_callback(self, base_lr, max_lr, step_size, cycle_mode, gamma=1):
        self.cycleLR = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.cycle_mode = cycle_mode
        self.gamma = gamma
        if self.cycle_mode == "exp_range":
            self.clr = CyclicLR(
                base_lr=self.base_lr,
                max_lr=self.max_lr,
                step_size=self.step_size,
                mode=self.cycle_mode,
                gamma=self.gamma,
            )
        self.callbacks.append(self.clr)

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

    def get_cycleLR(self):
        return self.cycleLR

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

    def get_lr_iterations(self):
        print(self.clr.history)
        return self.clr.history["lr"]

    # optimizer
    def set_optimizer(self, name="adam"):
        if name == "adam":
            self.optimizer = adam_optimizer()
        elif name == "SGD":
            self.optimizer = sgd_optimizer()

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
