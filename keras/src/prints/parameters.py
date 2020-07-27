"""
Copyright (c) 2019 CRISP

print info.

:author: Bahareh Tolooshams
"""

import yaml
import sys

PATH = sys.path[-1]


def print_model_info(folder_name):
    """
    prints model info
    :param folder_name located experiment folder
    :return: none
    """
    file = open(
        "{}/experiments/{}/config/config_model.yml".format(PATH, folder_name), "rb"
    )
    config_m = yaml.load(file)
    file.close()

    print("input_dim:", config_m["input_dim"])
    print("num_conv:", config_m["num_conv"])
    print("dictionary_dim:", config_m["dictionary_dim"])
    print("num_iterations:", config_m["num_iterations"])
    print("L:", config_m["L"])
    print("twosided:", config_m["twosided"])
    if "alpha" in config_m:
        print("alpha:", config_m["alpha"])
    print("data_space:", config_m["data_space"])
    if "lambda_trainable" in config_m:
        print("lambda_trainable:", config_m["lambda_trainable"])
        if config_m["lambda_trainable"]:
            print("lambda_EM:", config_m["lambda_EM"])
            print("lambda_single:", config_m["lambda_single"])


def print_training_info(folder_name):
    """
    prints training info
    :param folder_name located experiment folder
    :return: none
    """
    file = open(
        "{}/experiments/{}/config/config_model.yml".format(PATH, folder_name), "rb"
    )
    config_m = yaml.load(file)
    file.close()
    print("num_epochs:", config_m["num_epochs"])
    print("batch_size:", config_m["batch_size"])
    print("val_split:", config_m["val_split"])
    print("loss:", config_m["loss"])
    print("optimizer:", config_m["optimizer"])
    print("lr:", config_m["lr"])
    print("amsgrad:", config_m["amsgrad"])
    print("close:", config_m["close"])
    print("augment:", config_m["augment"])
