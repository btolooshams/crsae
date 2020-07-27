"""
Copyright (c) 2019 CRISP

functions to run experiment with CRsAE.

:author: Bahareh Tolooshams
"""
import os

# set this to the GPU name/number you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings

warnings.filterwarnings("ignore")

import click
import yaml
import h5py
import numpy as np
import sys
import itertools
import time
from time import gmtime, strftime
from keras.datasets import mnist
from keras.utils import np_utils

sys.path.append("..")
PATH = sys.path[-1]

from src.models.CRsAE import *
from src.models.LCSC import *
from src.models.TLAE import *
from src.prints.parameters import *
from src.plotter.plot_experiment_results import *
from src.run_experiments.extract_results_helpers import *


@click.group(chain=True)
def run_experiment():
    pass


@run_experiment.command()
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
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
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
            if "decay" in config_m:
                crsae.trainer.optimizer.set_beta_2(config_m["decay"])
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

        # add callbacks
        print("add callbacks.")
        crsae.trainer.add_best_val_loss_callback(config_m["loss_type"])
        crsae.trainer.add_all_epochs_callback(config_m["loss_type"])
        crsae.trainer.add_earlystopping_callback(
            config_m["min_delta"], config_m["patience"], config_m["loss_type"]
        )
        if config_m["cycleLR"]:
            if config_m["cycle_mode"] == "exp_range":
                crsae.trainer.add_cyclic_lr_callback(
                    config_m["base_lr"],
                    config_m["max_lr"],
                    config_m["step_size"],
                    config_m["cycle_mode"],
                    config_m["gamma"],
                )
    ################################################
    # build model knowing noiseSTD
    noiseSTD = 0.01
    crsae.build_model(noiseSTD)
    # initialize filter
    if crsae.trainer.get_close():
        H_init = np.load("{}/experiments/{}/data/H_init.npy".format(PATH, folder_name))
        crsae.set_H(H_init)
    ################################################
    # get initial H
    H_initial = crsae.get_H()
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    g_ch = hf.get("{}".format(config_d["ch"]))
    y_train = np.array(g_ch.get("y_train"))
    y_test = np.array(g_ch.get("y_test"))
    ################################################
    # get lambda
    print("lambda before training:", crsae.get_lambda())
    print("noiseSTD before training:", crsae.get_noiseSTD())
    ################################################
    # get initial H
    H_init = crsae.get_H()
    # get initial lambda
    lambda_init = crsae.get_lambda()
    ################################################
    # train
    time = crsae.train_and_save(y_train, folder_name)
    ################################################
    # get lambda
    print("lambda after training:", crsae.get_lambda())
    print("noiseSTD after training:", crsae.get_noiseSTD())
    ################################################
    # predict
    print("do prediciton.")
    z_test_hat = crsae.encode(y_test)
    y_test_hat = crsae.denoise(y_test)
    y_test_hat_separate = crsae.separate(y_test)
    ###############################################
    # save prediction
    print("save prediction results.")
    hf = h5py.File(
        "{}/experiments/{}/results/results_prediction_{}.h5".format(
            PATH, folder_name, time
        ),
        "w",
    )
    g_ch = hf.create_group("{}".format(config_d["ch"]))
    g_ch.create_dataset(
        "z_test_hat", data=z_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat", data=y_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat_separate",
        data=y_test_hat_separate,
        compression="gzip",
        compression_opts=9,
    )
    g_ch.create_dataset("H_init", data=H_initial)
    hf.close()


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def real_series(folder_name):
    # load model parameters
    print("load model parameters.")
    file = open("../experiments/{}/config/config_model.yml".format(folder_name), "rb")
    config_m = yaml.load(file)
    file.close()
    # load data parameters
    print("load data parameters.")
    file = open("../experiments/{}/config/config_data.yml".format(folder_name), "rb")
    config_d = yaml.load(file)
    file.close()
    ################################################
    # create CRsAE object
    print("create CRsAE object.")
    print_model_info(folder_name)
    if config_m["data_space"] == 1:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
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
        if "decay" in config_m:
            crsae.trainer.optimizer.set_beta_2(config_m["decay"])
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

    # add callbacks
    print("add callbacks.")
    crsae.trainer.add_best_val_loss_callback(config_m["loss_type"])
    crsae.trainer.add_all_epochs_callback(config_m["loss_type"])
    crsae.trainer.add_earlystopping_callback(
        config_m["min_delta"], config_m["patience"], config_m["loss_type"]
    )
    if config_m["cycleLR"]:
        if config_m["cycle_mode"] == "exp_range":
            crsae.trainer.add_cyclic_lr_callback(
                config_m["base_lr"],
                config_m["max_lr"],
                config_m["step_size"],
                config_m["cycle_mode"],
                config_m["gamma"],
            )
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("../experiments/{}/data/data.h5".format(folder_name), "r")
    g_ch = hf.get("{}".format(config_d["ch"]))
    y_train = np.array(g_ch.get("y_train"))
    y_test = np.array(g_ch.get("y_test"))
    y_series = np.array(g_ch.get("y_series"))
    length_of_data = np.array(g_ch.get("length_of_data"))
    noiseSTD = np.array(g_ch.get("noiseSTD"))
    print("noiseSTD:", noiseSTD)
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    # initialize filter
    if crsae.trainer.get_close():
        H_init = np.load("../experiments/{}/data/H_init.npy".format(folder_name))
        crsae.set_H(H_init)
    ################################################
    # build model for series prediction
    # create CRsAE object
    print_model_info(folder_name)
    if config_m["data_space"] == 1:
        if config_m["lambda_trainable"]:
            crsae_series = CRsAE_1d(
                length_of_data,
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae_series = CRsAE_1d(
                length_of_data,
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae_series = CRsAE_2d(
                length_of_data,
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae_series = CRsAE_2d(
                length_of_data,
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    # build model knowing noiseSTD
    crsae_series.build_model(noiseSTD)
    ################################################
    # get lambda
    print("lambda before training:", crsae.get_lambda())
    print("noiseSTD before training:", crsae.get_noiseSTD())
    ################################################
    # get initial H
    H_init = crsae.get_H()
    # get initial lambda
    lambda_init = crsae.get_lambda()
    ################################################
    # train
    time = crsae.train_and_save(y_train, folder_name)
    ################################################
    # get lambda
    print("lambda after training:", crsae.get_lambda())
    print("noiseSTD after training:", crsae.get_noiseSTD())
    ################################################
    # predict
    print("do prediciton.")
    if (config_d["num_test"]) != 0:
        z_test_hat = crsae.encode(y_test)
        y_test_hat = crsae.denoise(y_test)
        y_test_hat_separate = crsae.separate(y_test)
    else:
        z_test_hat = crsae.encode(y_train)
        y_test_hat = crsae.denoise(y_train)
        y_test_hat_separate = crsae.separate(y_train)

    H_learned = crsae.get_H()
    lambda_learned = crsae.get_lambda()
    crsae_series.set_H(H_learned)
    crsae_series.set_lambda(lambda_learned)
    z_series_hat = crsae_series.encode(y_series)
    y_series_hat = crsae_series.denoise(y_series)
    y_series_hat_separate = crsae_series.separate(y_series)
    ###############################################
    # save prediction
    print("save prediction results.")
    hf = h5py.File(
        "../experiments/{}/results/results_prediction_{}.h5".format(folder_name, time),
        "w",
    )
    g_ch = hf.create_group("{}".format(config_d["ch"]))
    g_ch.create_dataset(
        "z_test_hat", data=z_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat", data=y_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "z_series_hat", data=z_series_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_series_hat", data=y_series_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat_separate",
        data=y_test_hat_separate,
        compression="gzip",
        compression_opts=9,
    )
    g_ch.create_dataset(
        "y_series_hat_separate",
        data=y_series_hat_separate,
        compression="gzip",
        compression_opts=9,
    )
    g_ch.create_dataset("H_init", data=H_init)
    hf.close()


@run_experiment.command()
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
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
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
        if "decay" in config_m:
            crsae.trainer.optimizer.set_beta_2(config_m["decay"])
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

    # add callbacks
    print("add callbacks.")
    crsae.trainer.add_best_val_loss_callback(config_m["loss_type"])
    crsae.trainer.add_all_epochs_callback(config_m["loss_type"])
    crsae.trainer.add_earlystopping_callback(
        config_m["min_delta"], config_m["patience"], config_m["loss_type"]
    )
    if config_m["cycleLR"]:
        if config_m["cycle_mode"] == "exp_range":
            crsae.trainer.add_cyclic_lr_callback(
                config_m["base_lr"],
                config_m["max_lr"],
                config_m["step_size"],
                config_m["cycle_mode"],
                config_m["gamma"],
            )
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
            if np.min(dist_true_learned) >= 0.4:
                if np.max(dist_true_learned) <= 0.5:
                    flag = 0

        # # this is temp
        # H_init = np.load(
        #     "{}/experiments/{}/data/H_init.npy".format(PATH, folder_name)
        # )
        # crsae.set_H(H_init)
        dist_true_learned, temp = get_err_h1_h2(H_true, crsae.get_H())
        print("initial distance err:", dist_true_learned)
    ################################################
    z_test = np.array(hf.get("z_test"))
    l1_norm_z_test = np.mean(np.sum(np.sum(np.abs(z_test), axis=2), axis=1), axis=0)
    print("l1_norm from true code:", l1_norm_z_test)
    lambda_estimate = (
        (config_m["input_dim"] - config_m["dictionary_dim"] + 1) * config_m["num_conv"]
    ) / l1_norm_z_test
    print("lambda estimate from true code:", lambda_estimate)

    y_test = np.array(hf.get("y_test"))
    z_test_from_FISTA = crsae.encode(y_test)
    l1_norm_z_test_from_FISTA = np.mean(
        np.sum(np.sum(np.abs(z_test_from_FISTA), axis=2), axis=1), axis=0
    )
    print("l1_norm from code through FISTA:", l1_norm_z_test_from_FISTA)
    lambda_estimate_from_FISTA = (
        (config_m["input_dim"] - config_m["dictionary_dim"] + 1) * config_m["num_conv"]
    ) / l1_norm_z_test_from_FISTA
    print("lambda estimate from code through FISTA:", lambda_estimate_from_FISTA)
    ################################################

    # get lambda
    print("lambda before training:", crsae.get_lambda())
    print("noiseSTD before training:", crsae.get_noiseSTD())
    ################################################
    # get initial H
    H_init = crsae.get_H()
    # get initial lambda
    lambda_init = crsae.get_lambda()
    ################################################
    # train
    time = crsae.train_and_save(y_train_noisy, folder_name)
    ################################################
    # # save results
    # time = crsae.save_results(folder_name)
    ################################################
    # get lambda
    print("lambda after training:", crsae.get_lambda())
    print("noiseSTD after training:", crsae.get_noiseSTD())
    ################################################
    # predict
    print("do prediciton.")
    z_test_hat = crsae.encode(y_test_noisy)
    y_test_hat = crsae.denoise(y_test_noisy)
    y_test_hat_separate = crsae.separate(y_test_noisy)
    ###############################################
    # save prediction
    print("save prediction results.")
    hf = h5py.File(
        "{}/experiments/{}/results/results_prediction_{}.h5".format(
            PATH, folder_name, time
        ),
        "w",
    )
    g_ch = hf.create_group("{}".format(config_d["ch"]))
    g_ch.create_dataset(
        "z_test_hat", data=z_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat", data=y_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset(
        "y_test_hat_separate",
        data=y_test_hat_separate,
        compression="gzip",
        compression_opts=9,
    )
    g_ch.create_dataset("H_init", data=H_init)
    g_ch.create_dataset("lambda_init", data=lambda_init)
    hf.close()


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def lcsc_simulated(folder_name):
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
    for p in range(10):
        # create CRsAE object
        print("create CRsAE object.")
        print_model_info(folder_name)
        if config_m["data_space"] == 1:
            lcsc = LCSC_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
            )
        else:
            print("ERROR: 2D version of LCSC is not implemented.")
        ################################################
        # configure trainer
        print("configure trainer.")
        # trainer parameters
        print_training_info(folder_name)
        lcsc.trainer.set_num_epochs(config_m["num_epochs"])
        lcsc.trainer.set_batch_size(config_m["batch_size"])
        lcsc.trainer.set_verbose(config_m["verbose"])
        lcsc.trainer.set_val_split(config_m["val_split"])
        lcsc.trainer.set_loss(config_m["loss"])
        lcsc.trainer.set_close(config_m["close"])
        lcsc.trainer.set_augment(config_m["augment"])
        # optimizer
        lcsc.trainer.set_optimizer(config_m["optimizer"])
        lcsc.trainer.optimizer.set_lr(config_m["lr"])
        lcsc.trainer.optimizer.set_amsgrad(config_m["amsgrad"])
        # add callbacks
        print("add callbacks.")
        lcsc.trainer.add_best_val_loss_callback(config_m["loss_type"])
        lcsc.trainer.add_all_epochs_callback(config_m["loss_type"])
        lcsc.trainer.add_earlystopping_callback(
            config_m["min_delta"], config_m["patience"], config_m["loss_type"]
        )
        if config_m["cycleLR"]:
            lcsc.trainer.add_cyclic_lr_callback(
                config_m["base_lr"], config_m["max_lr"], config_m["step_size"]
            )
        ################################################
        # load data
        print("load data.")
        hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
        y_train_noisy = np.array(hf.get("y_train_noisy"))
        y_test_noisy = np.array(hf.get("y_test_noisy"))
        noiseSTD = np.array(hf.get("noiseSTD"))
        print("noiseSTD:", noiseSTD)
        ################################################
        # build model
        lcsc.build_model()
        # initialize filter
        if lcsc.trainer.get_close():
            We_true = np.load(
                "{}/experiments/{}/data/H_true.npy".format(PATH, folder_name)
            )
            weights_noisestd = 0.5 * np.std(np.squeeze(We_true), axis=0)
            print(weights_noisestd)
            We_noisy = np.copy(We_true)
            for n in range(config_m["num_conv"]):
                We_noisy[:, :, n] += weights_noisestd[n] * np.random.randn(
                    lcsc.get_dictionary_dim(), 1
                )

            Wd_noisy = np.expand_dims(np.flip(np.squeeze(We_noisy), axis=0), axis=2)
            d_noisy = np.expand_dims(np.flip(np.squeeze(We_noisy), axis=0), axis=2)
            We_noisy /= config_m["L"]
            lcsc.set_weights(Wd_noisy, We_noisy, d_noisy)
        ################################################
        if config_m["lambda_trainable"]:
            lcsc.set_lambda(np.zeros(config_m["num_conv"]) + (1 / config_m["L"]))
        else:
            donoho_estimate = noiseSTD * np.sqrt(
                2
                * np.log(
                    config_m["num_conv"]
                    * (config_m["input_dim"] - config_m["dictionary_dim"] + 1)
                )
            )
            lcsc.set_lambda(np.zeros(config_m["num_conv"]) + donoho_estimate)

        ################################################
        # get lambda
        print("lambda (regulariztion parameter) before training:", lcsc.get_lambda())
        ################################################
        # get initial weights
        Wd_initial = lcsc.get_Wd()
        We_initial = lcsc.get_We()
        d_initial = lcsc.get_d()
        lambda_initial = lcsc.get_lambda()
        ################################################
        # train
        time = lcsc.train_and_save(y_train_noisy, folder_name)
        ################################################
        # # save results
        # time = crsae.save_results(folder_name)
        ################################################
        # get lambda
        print("lambda (regulariztion parameter) after training:", lcsc.get_lambda())
        ################################################
        # predict
        print("do prediciton.")
        z_test_hat = lcsc.encode(y_test_noisy)
        y_test_hat = lcsc.denoise(y_test_noisy)
        y_test_hat_separate = lcsc.separate(y_test_noisy)
        ###############################################
        # save prediction
        print("save prediction results.")
        hf = h5py.File(
            "{}/experiments/{}/results/LCSC_results_prediction_{}.h5".format(
                PATH, folder_name, time
            ),
            "w",
        )
        g_ch = hf.create_group("{}".format(config_d["ch"]))
        g_ch.create_dataset(
            "z_test_hat", data=z_test_hat, compression="gzip", compression_opts=9
        )
        g_ch.create_dataset(
            "y_test_hat", data=y_test_hat, compression="gzip", compression_opts=9
        )
        g_ch.create_dataset(
            "y_test_hat_separate",
            data=y_test_hat_separate,
            compression="gzip",
            compression_opts=9,
        )
        g_ch.create_dataset("Wd_init", data=Wd_initial)
        g_ch.create_dataset("We_init", data=We_initial)
        g_ch.create_dataset("d_init", data=d_initial)
        g_ch.create_dataset("lambda_init", data=lambda_initial)
        hf.close()


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def tlae_simulated(folder_name):
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
        if config_m["lambda_trainable"]:
            crsae = TLAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = TLAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = TLAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = TLAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
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
    if config_m["lambda_trainable"]:
        crsae.trainer.optimizer.set_lambda_lr(config_m["lambda_lr"])
    crsae.trainer.optimizer.set_amsgrad(config_m["amsgrad"])
    # add callbacks
    print("add callbacks.")
    crsae.trainer.add_best_val_loss_callback(config_m["loss_type"])
    crsae.trainer.add_all_epochs_callback(config_m["loss_type"])
    crsae.trainer.add_earlystopping_callback(
        config_m["min_delta"], config_m["patience"], config_m["loss_type"]
    )
    if config_m["cycleLR"]:
        crsae.trainer.add_cyclic_lr_callback(
            config_m["base_lr"], config_m["max_lr"], config_m["step_size"]
        )
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
        # H_noisestd = 0
        print(H_noisestd)
        crsae.set_H(H_true, H_noisestd)
    ################################################
    # get lambda
    print("lambda before training:", crsae.get_lambda())
    ################################################
    # get initial H
    H_init = crsae.get_H()
    # get initial lambda
    lambda_init = crsae.get_lambda()
    ################################################
    # train
    time = crsae.train_and_save(y_train_noisy, folder_name)
    ################################################
    # # save results
    # time = crsae.save_results(folder_name)
    ################################################
    # get lambda
    print("lambda after training:", crsae.get_lambda())
    ################################################
    # predict
    print("do prediciton.")
    z_test_hat = crsae.encode(y_test_noisy)
    ###############################################
    # save prediction
    print("save prediction results.")
    hf = h5py.File(
        "{}/experiments/{}/results/TLAE_results_prediction_{}.h5".format(
            PATH, folder_name, time
        ),
        "w",
    )
    g_ch = hf.create_group("{}".format(config_d["ch"]))
    g_ch.create_dataset(
        "z_test_hat", data=z_test_hat, compression="gzip", compression_opts=9
    )
    g_ch.create_dataset("H_init", data=H_init)
    g_ch.create_dataset("lambda_init", data=lambda_init)
    hf.close()


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def simulated_fista_iteration_test(folder_name):
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
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    y_test_noisy = np.array(hf.get("y_test_noisy"))
    z_test = np.array(hf.get("z_test"))
    noiseSTD = np.array(hf.get("noiseSTD"))
    print("noiseSTD:", noiseSTD)
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    H_true = np.load("{}/experiments/{}/data/H_true.npy".format(PATH, folder_name))
    crsae.set_H(H_true, 0)

    z_test_hat = crsae.encode(y_test_noisy)

    best_permutation_index = 0
    file_number = config_m["num_iterations"]
    plot_code_sim(
        8,
        z_test,
        z_test_hat,
        best_permutation_index,
        PATH,
        folder_name,
        file_number,
        config_d["sampling_rate"],
        row=1,
        line_width=2,
        marker_size=15,
        scale=4,
        scale_height=0.5,
        text_font=45,
        title_font=45,
        axes_font=48,
        legend_font=32,
        number_font=40,
    )


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def simulated_csc_speed(folder_name):
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
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["delta"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["alpha"],
                config_m["num_channels"],
            )
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    y_train_noisy = np.array(hf.get("y_train_noisy"))
    noiseSTD = np.array(hf.get("noiseSTD"))
    print("noiseSTD:", noiseSTD)
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    H_true = np.load("{}/experiments/{}/data/H_true.npy".format(PATH, folder_name))
    crsae.set_H(H_true, 0)

    z_train_hat = crsae.encode(y_train_noisy)
    csc_times = []
    for k in range(50):
        csc_start_time = time.time()
        z_train_hat = crsae.encode(y_train_noisy)
        csc_times.append(time.time() - csc_start_time)
    csc_time = np.mean(csc_times)

    print(
        "csc time: {} s for {} examples each {} length".format(
            csc_time, y_train_noisy.shape[0], y_train_noisy.shape[1]
        )
    )


@run_experiment.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def simulated_check_speed(folder_name):
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
        if config_m["lambda_trainable"]:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_1d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["noiseSTD_lr"],
            )
    else:
        if config_m["lambda_trainable"]:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["lambda_EM"],
                config_m["delta"],
                config_m["lambda_single"],
                config_m["noiseSTD_lr"],
            )
        else:
            crsae = CRsAE_2d(
                config_m["input_dim"],
                config_m["num_conv"],
                config_m["dictionary_dim"],
                config_m["num_iterations"],
                config_m["L"],
                config_m["twosided"],
                config_m["lambda_trainable"],
                config_m["alpha"],
                config_m["num_channels"],
                config_m["noiseSTD_trainable"],
                config_m["noiseSTD_lr"],
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
        if "decay" in config_m:
            crsae.trainer.optimizer.set_beta_2(config_m["decay"])
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

    # add callbacks
    print("add callbacks.")
    crsae.trainer.add_best_val_loss_callback(config_m["loss_type"])
    crsae.trainer.add_all_epochs_callback(config_m["loss_type"])
    crsae.trainer.add_earlystopping_callback(
        config_m["min_delta"], config_m["patience"], config_m["loss_type"]
    )
    if config_m["cycleLR"]:
        if config_m["cycle_mode"] == "exp_range":
            crsae.trainer.add_cyclic_lr_callback(
                config_m["base_lr"],
                config_m["max_lr"],
                config_m["step_size"],
                config_m["cycle_mode"],
                config_m["gamma"],
            )
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
    # train
    time = crsae.check_speed(y_train_noisy)
    print(time)


if __name__ == "__main__":
    run_experiment()
