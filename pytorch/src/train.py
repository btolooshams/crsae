"""
Copyright (c) 2020 CRISP

train

:author: Bahareh Tolooshams
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sparselandtools.dictionaries import DCTDictionary
import os
from tqdm import tqdm
from datetime import datetime
from sacred import Experiment

from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from scipy.special import expit
from pytorch_msssim import MS_SSIM

import sys

sys.path.append("src/")

import model, generator, trainer, utils, conf

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")


ex = Experiment("train", ingredients=[config_ingredient])


@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    print("load data.")
    if hyp["dataset"] == "path":
        train_loader = generator.get_path_loader(
            hyp["batch_size"],
            hyp["train_path"],
            shuffle=hyp["shuffle"],
            crop_dim=hyp["crop_dim"],
        )
        test_loader = generator.get_path_loader(1, hyp["test_path"], shuffle=False)
    if hyp["dataset"] == "VOC":
        train_loader, _ = generator.get_VOC_loaders(
            hyp["batch_size"],
            crop_dim=hyp["crop_dim"],
            shuffle=hyp["shuffle"],
            image_set=hyp["image_set"],
            segmentation=hyp["segmentation"],
            year=hyp["year"],
        )
        test_loader = generator.get_path_loader(1, hyp["test_path"], shuffle=False)
    else:
        print("dataset is not implemented.")

    if hyp["init_with_DCT"]:
        dct_dictionary = DCTDictionary(
            hyp["dictionary_dim"], np.int(np.sqrt(hyp["num_conv"]))
        )
        H_init = dct_dictionary.matrix.reshape(
            hyp["dictionary_dim"], hyp["dictionary_dim"], hyp["num_conv"]
        ).T
        H_init = np.expand_dims(H_init, axis=1)
        H_init = torch.from_numpy(H_init).float().to(hyp["device"])
    else:
        H_init = None

    print("create model.")
    if hyp["network"] == "CRsAE1D":
        net = model.CRsAE1D(hyp, H_init)
    elif hyp["network"] == "CRsAE1DTrainableBias":
        net = model.CRsAE1DTrainableBias(hyp, H_init)
    elif hyp["network"] == "CRsAE2D":
        net = model.CRsAE2D(hyp, H_init)
    elif hyp["network"] == "CRsAE2DFreeBias":
        net = model.CRsAE2DFreeBias(hyp, H_init)
    elif hyp["network"] == "CRsAE2DUntied":
        net = model.CRsAE2DUntied(hyp, H_init)
    elif hyp["network"] == "CRsAE2DUntiedFreeBias":
        net = model.CRsAE2DUntiedFreeBias(hyp, H_init)
    elif hyp["network"] == "CRsAE2DTrainableBias":
        net = model.CRsAE2DTrainableBias(hyp, H_init)
    elif hyp["network"] == "CRsAE2DUntiedTrainableBias":
        net = model.CRsAE2DUntiedTrainableBias(hyp, H_init)
    else:
        print("model does not exist!")

    torch.save(net, os.path.join(PATH, "model_init.pt"))

    if hyp["trainable_bias"]:
        if hyp["loss"] == "MSE":
            criterion_ae = torch.nn.MSELoss()
        elif hyp["loss"] == "L1":
            criterion_ae = torch.nn.L1Loss()
        elif hyp["loss"] == "MSSSIM_l1":
            criterion_ae = utils.MSSSIM_l1()
        criterion_lam = utils.LambdaLoss2D()

        param_ae = []
        param_lam = []
        ctr = 0
        if hyp["network"] == "CRsAE2DUntiedTrainableBias":
            a = 3
        else:
            a = 1
        for param in net.parameters():

            if ctr == a:
                param_lam.append(param)
                print("lam", param.shape)
            else:
                param_ae.append(param)
                print("ae", param.shape)

            ctr += 1

        optimizer_ae = optim.Adam(param_ae, lr=hyp["lr"], eps=1e-3)
        optimizer_lam = optim.Adam(param_lam, lr=hyp["lr_lam"], eps=1e-3)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer_ae, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
        )
    else:
        if hyp["loss"] == "MSE":
            criterion = torch.nn.MSELoss()
        elif hyp["loss"] == "L1":
            criterion = torch.nn.L1Loss()
        elif hyp["loss"] == "MSSSIM_l1":
            criterion = utils.MSSSIM_l1()
        optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)

        if hyp["cyclic"]:
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=hyp["base_lr"],
                max_lr=hyp["max_lr"],
                step_size_up=hyp["step_size"],
                cycle_momentum=False,
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
            )

    print("train auto-encoder.")
    if hyp["trainable_bias"]:
        net = trainer.train_ae_withtrainablebias(
            net,
            train_loader,
            hyp,
            criterion_ae,
            criterion_lam,
            optimizer_ae,
            optimizer_lam,
            scheduler,
            PATH,
            test_loader,
            0,
            hyp["num_epochs"],
        )
    else:
        net = trainer.train_ae(
            net,
            train_loader,
            hyp,
            criterion,
            optimizer,
            scheduler,
            PATH,
            test_loader,
            0,
            hyp["num_epochs"],
        )

    print("training finished!")
