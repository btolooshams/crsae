"""
Copyright (c) 2019 CRISP

classes related to ADAM optimizer, etc.

:author: Bahareh Tolooshams
"""

from keras.optimizers import Adam


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
