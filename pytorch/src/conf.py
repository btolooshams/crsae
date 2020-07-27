"""
Copyright (c) 2019 CRISP

config

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "default",
        "dataset": "MNIST",
        "network": "CRsAE2D",
        "dictionary_dim": 5,
        "num_conv": 64,
        "stride": 6,
        "L": 100,
        "trainable_bias": False,
        "delta": 50,
        "num_iters": 50,
        "batch_size": 16,
        "num_epochs": 20,
        "normalize": True,
        "lr": 5e-4,
        "lr_decay": 0.7,
        "lr_step": 10,
        "lr_lam": 1e-2,
        "cyclic": False,
        "noiseSTD": 20,
        "shuffle": True,
        "test_path": "../data/test_img/",
        "train_path": "../data/test_img/",
        "device": device,
        "warm_start": True,
        "info_period": 10,
        "denoising": True,
        "supervised": True,
        "crop_dim": (250, 250),
        "init_with_DCT": True,
        "init_with_saved_file": False,
        "sigma": 0.18,
        "loss": "MSE",
        "lam": 0.1,
        "twosided": True,
        "image_set": "train",
        "year": "2012",
        "segmentation": True,
    }


@config_ingredient.named_config
def crsae_msssim():
    hyp = {
        "experiment_name": "crsae_msssim",
        "dataset": "VOC",
        "image_set": "train",
        "year": "2012",
        "segmentation": False,
        "network": "CRsAE2DUntiedTrainableBias",
        "dictionary_dim": 7,
        "stride": 5,
        "num_conv": 64,
        "L": 10,
        "num_iters": 30,
        "batch_size": 1,
        "num_epochs": 300,
        "normalize": True,
        "lr": 0.01,
        "lr_lam": 0.1,
        "lr_decay": 0.7,
        "lr_step": 10,
        "noiseSTD": 20,
        "sigma": 0.078,
        "shuffle": True,
        "test_path": "../data/test_img/",
        "info_period": 2000,
        "denoising": True,
        "supervised": True,
        "crop_dim": (128, 128),
        "init_with_DCT": False,
        "init_with_saved_file": False,
        "loss": "MSSSIM_l1",
        "trainable_bias": True,
        "lam": 20,
        "delta": 100,
    }
