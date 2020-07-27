"""
Copyright (c) 2019 CRISP

classes related to SGD optimizer, etc.

:author: Bahareh Tolooshams
"""

from keras.optimizers import SGD


class sgd_optimizer:
    def __init__(
        self, lr=0.001, momentum=0.0, decay=0.0, nesterov=False, lambda_lr=0.001
    ):
        self.name = "sgd"
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.lambda_lr = lambda_lr

        self.update_optimizer()

    def update_optimizer(self):
        self.keras_optimizer = SGD(
            lr=self.lr, momentum=self.momentum, decay=self.decay, nesterov=self.nesterov
        )
        self.keras_optimizer_for_lambda = SGD(
            lr=self.lambda_lr,
            momentum=self.momentum,
            decay=self.decay,
            nesterov=self.nesterov,
        )

    def set_lr(self, lr):
        self.lr = lr
        self.update_optimizer()

    def set_lambda_lr(self, lambda_lr):
        self.lambda_lr = lambda_lr
        self.update_optimizer()

    def set_momentum(self, momentum):
        self.momentum = momentum
        self.update_optimizer()

    def set_decay(self, decay):
        self.decay = decay
        self.update_optimizer()

    def set_nesterov(self, nesterov):
        self.nesterov = nesterov
        self.update_optimizer()

    def get_name(self):
        return self.name

    def get_lr(self):
        return self.lr

    def get_lambda_lr(self):
        return self.lambda_lr

    def get_momentum(self):
        return self.momentum

    def get_decay(self):
        return self.decay

    def get_nesterov(self):
        return self.nesterov

    def get_keras_optimizer(self):
        return self.keras_optimizer

    def get_keras_optimizer_for_lambda(self):
        return self.keras_optimizer_for_lambda
