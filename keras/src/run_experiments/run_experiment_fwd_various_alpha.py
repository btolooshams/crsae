"""
Copyright (c) 2019 CRISP

functions to run experiment with CRsAE.

:author: Bahareh Tolooshams
"""
import os

# set this to the GPU name/number you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings

warnings.filterwarnings("ignore")

import click
import yaml
import h5py
import numpy as np
import sys

sys.path.append("..")
PATH = sys.path[-1]

from src.models.CRsAE import *
from src.prints.parameters import *


@click.group(chain=True)
def run_experiment_fwd_various_alpha():
    pass


@run_experiment_fwd_various_alpha.command()
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
            config_m["lambda_trainable"],
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
            config_m["alpha"],
            config_m["num_channels"],
        )
    ################################################
    # load data
    print("load data.")
    hf = h5py.File("{}/experiments/{}/data/data.h5".format(PATH, folder_name), "r")
    y_train_noisy = np.array(hf.get("y_train_noisy"))
    y_test_noisy = np.array(hf.get("y_test_noisy"))
    noiseSTD = np.array(hf.get("noiseSTD"))
    print("noiseSTD:", noiseSTD)
    H_true = np.load("{}/experiments/{}/data/H_true.npy".format(PATH, folder_name))
    ################################################
    # build model knowing noiseSTD
    crsae.build_model(noiseSTD)
    # set filters
    crsae.set_H(H_true)
    ################################################
    # get lambda
    lambda_donoho = crsae.get_lambda()
    print("lambda (regulariztion parameter):", crsae.get_lambda())
    ################################################
    # predict
    print("do prediciton.")
    alpha_list = np.arange(0.1, 2.05, 0.05)
    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    hf = h5py.File(
        "{}/experiments/{}/results/results_fwd_{}.h5".format(PATH, folder_name, time),
        "w",
    )
    g_ch = hf.create_group("{}".format(config_d["ch"]))
    g_ch.create_dataset(
        "alpha_list", data=alpha_list, compression="gzip", compression_opts=9
    )
    ctr = 0
    for alpha in alpha_list:
        print("alpha:", alpha)
        crsae.set_lambda(lambda_donoho * alpha)
        y_test_hat = crsae.denoise(y_test_noisy)

        g_ch.create_dataset(
            "y_test_hat_{}".format(ctr),
            data=y_test_hat,
            compression="gzip",
            compression_opts=9,
        )
        ctr += 1
    hf.close()


if __name__ == "__main__":
    run_experiment_fwd_various_alpha()
