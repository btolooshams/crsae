"""
Copyright (c) 2019 CRISP

functions to run experiment with CRsAE to find lr, etc.

:author: Bahareh Tolooshams
"""
import os

# set this to the GPU name/number you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings

warnings.filterwarnings("ignore")

import yaml
import h5py
import sys
import numpy as np
import click

sys.path.append("..")
PATH = sys.path[-1]

from src.models.CRsAE import *
from src.prints.parameters import *
from src.run_experiments.extract_results_helpers import *


@click.group(chain=True)
def run_experiment_find_lr():
    pass


@run_experiment_find_lr.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def real(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open(
        "{}/experiments/{}/config/config_model.yml".format(PATH, folder_name), "rb"
    )
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    ################################################
    # create CRsAE object
    print("create CRsAE object.")
    print_model_info(folder_name)
    if config_m["data_space"] == 1:
        crsae = CRsAE_1d(
            config_m["input_dim"],
            config_m["num_conv"],
            config_m["dictionary_dim"],
            config_m["num_iterations"],
            config_m["L"],
            config_m["twosided"],
            False,
            config_m["alpha"],
            config_m["num_channels"],
        )
    else:
        crsae = CRsAE_2d(
            config_m["input_dim"],
            config_m["num_conv"],
            config_m["dictionary_dim"],
            config_m["num_iterations"],
            config_m["L"],
            config_m["twosided"],
            False,
            config_m["alpha"],
            config_m["num_channels"],
        )
    ################################################
    # configure trainer
    print("configure trainer.")
    # trainer parameters
    print_training_info(folder_name)
    crsae.trainer.set_num_epochs(config_m["num_epochs"])
    crsae.trainer.set_batch_size(config_m["batch_size"])
    crsae.trainer.set_verbose(config_m["verbose"])
    crsae.trainer.set_val_split(config_m["val_split"])
    crsae.trainer.set_loss(config_m["loss"])
    crsae.trainer.set_close(config_m["close"])
    crsae.trainer.set_augment(config_m["augment"])
    # optimizer
    crsae.trainer.set_optimizer(config_m["optimizer"])
    crsae.trainer.optimizer.set_lr(config_m["lr"])
    # ADAM
    if config_m["optimizer"] == "Adam":
        if "beta_1" in config_m:
            crsae.trainer.optimizer.set_beta_1(config_m["beta_1"])
        if "beta_2" in config_m:
            crsae.trainer.optimizer.set_beta_2(config_m["beta_2"])
        crsae.trainer.optimizer.set_amsgrad(config_m["amsgrad"])
    # SGD
    elif config_m["optimizer"] == "SGD":
        if "momentum" in config_m:
            crsae.trainer.optimizer.set_momentum(config_m["momentum"])
        if "decay" in config_m:
            crsae.trainer.optimizer.set_decay(config_m["decay"])
        if "nesterov" in config_m:
            crsae.trainer.optimizer.set_nesterov(config_m["nesterov"])
    if config_m["lambda_trainable"]:
        crsae.trainer.optimizer.set_lambda_lr(config_m["lambda_lr"])
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    g_ch = hf.get("{}".format(config_d["ch"]))
    y_train = np.array(g_ch.get("y_train"))
    noiseSTD = np.array(g_ch.get("noiseSTD"))
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    ################################################
    # initialize filter
    if crsae.trainer.get_close():
        H_init = np.load("{}/experiments/{}/data/H_init.npy".format(PATH, folder_name))
        crsae.set_H(H_init)
    ################################################
    # find_lr
    crsae.find_lr(y_train, folder_name, num_epochs=2)


@run_experiment_find_lr.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def simulated(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open(
        "{}/experiments/{}/config/config_model.yml".format(PATH, folder_name), "rb"
    )
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open(
        "{}/experiments/{}/config/config_data.yml".format(PATH, folder_name), "rb"
    )
    config_d = yaml.load(file)
    file.close()
    ################################################
    # create CRsAE object
    print("create CRsAE object.")
    print_model_info(folder_name)
    if config_m["data_space"] == 1:
        crsae = CRsAE_1d(
            config_m["input_dim"],
            config_m["num_conv"],
            config_m["dictionary_dim"],
            config_m["num_iterations"],
            config_m["L"],
            config_m["twosided"],
            False,
            config_m["alpha"],
            config_m["num_channels"],
        )
    else:
        crsae = CRsAE_2d(
            config_m["input_dim"],
            config_m["num_conv"],
            config_m["dictionary_dim"],
            config_m["num_iterations"],
            config_m["L"],
            config_m["twosided"],
            False,
            config_m["alpha"],
            config_m["num_channels"],
        )
    ################################################
    # configure trainer
    print("configure trainer.")
    # trainer parameters
    print_training_info(folder_name)
    crsae.trainer.set_num_epochs(config_m["num_epochs"])
    crsae.trainer.set_batch_size(config_m["batch_size"])
    crsae.trainer.set_verbose(config_m["verbose"])
    crsae.trainer.set_val_split(config_m["val_split"])
    crsae.trainer.set_loss(config_m["loss"])
    crsae.trainer.set_close(config_m["close"])
    crsae.trainer.set_augment(config_m["augment"])
    # optimizer
    crsae.trainer.set_optimizer(config_m["optimizer"])
    crsae.trainer.optimizer.set_lr(config_m["lr"])
    # ADAM
    if config_m["optimizer"] == "Adam":
        if "beta_1" in config_m:
            crsae.trainer.optimizer.set_beta_1(config_m["beta_1"])
        if "beta_2" in config_m:
            crsae.trainer.optimizer.set_beta_2(config_m["beta_2"])
        crsae.trainer.optimizer.set_amsgrad(config_m["amsgrad"])
    # SGD
    elif config_m["optimizer"] == "SGD":
        if "momentum" in config_m:
            crsae.trainer.optimizer.set_momentum(config_m["momentum"])
        if "decay" in config_m:
            crsae.trainer.optimizer.set_decay(config_m["decay"])
        if "nesterov" in config_m:
            crsae.trainer.optimizer.set_nesterov(config_m["nesterov"])
    if config_m["lambda_trainable"]:
        crsae.trainer.optimizer.set_lambda_lr(config_m["lambda_lr"])
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    y_train_noisy = np.array(hf.get("y_train_noisy"))
    y_test_noisy = np.array(hf.get("y_test_noisy"))
    noiseSTD = np.array(hf.get("noiseSTD"))
    print("noiseSTD:", noiseSTD)
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    # initialize filter
    if crsae.trainer.get_close():
        H_true = np.load("{}/experiments/{}/data/H_true.npy".format(PATH, folder_name))
        H_noisestd = 0.5 * np.std(np.squeeze(H_true), axis=0)
        print(H_noisestd)
        flag = 1
        while flag:
            crsae.set_H(H_true, H_noisestd)
            dist_true_learned, temp = get_err_h1_h2(H_true, crsae.get_H())
            if np.max(dist_true_learned) <= 0.5:
                flag = 0
        print("initial distance err:", dist_true_learned)
    ################################################
    # find_lr
    crsae.find_lr(y_train_noisy, folder_name, num_epochs=2)


if __name__ == "__main__":
    run_experiment_find_lr()
