"""
Copyright (c) 2019 CRISP

Generate data

:author: Bahareh Tolooshams
"""
import sys

sys.path.append("..")
PATH = sys.path[-1]
import numpy as np
import scipy as sc
from scipy import stats
from src.models.single_layer_autoencoders import *
from keras.layers import Input, Conv2D, ZeroPadding2D
from keras.models import Model
import scipy.io as sio
import h5py
import click
import yaml


@click.group(chain=True)
def make_dataset():
    pass


# generate waveforms (same waveform has non-overlapping constraint)
# amplitude of waveforms are normally distributed
def generateDataNonOverlapping(
    HT, input_dim, dictionary_dim, num_conv, num_train, num_test, mean_std, density
):
    z_dim = input_dim - dictionary_dim + 1
    num_data = num_train + num_test

    # create decoder (data_generator)
    data_generator = decoder_1d(z_dim, dictionary_dim, num_conv)

    # generate sparse codes
    Z = np.zeros((num_data, z_dim, num_conv))
    for i in range(num_data):
        for n in range(num_conv):
            # rvs = stats.uniform(amp_range[0],amp_range[1]-amp_range[0]).rvs
            rvs = stats.norm(loc=mean_std[n, 0], scale=mean_std[n, 1]).rvs
            temp = sc.sparse.random(1, z_dim, density[n], data_rvs=rvs).A
            sign = np.concatenate(
                (np.ones((temp.shape[1], 1)), -np.ones((temp.shape[1], 1))), axis=0
            )
            np.random.shuffle(sign)
            Z[i, :, n] = temp * sign[0 : temp.shape[1], 0]
            indices = np.where(Z[i, :, n] != 0)[0]
            diff_indices = np.diff(indices)
            condition = np.sum(np.where(diff_indices <= dictionary_dim)[0] + 1)
            while condition != 0:
                for l in range(len(indices) - 1):
                    if diff_indices[l] <= dictionary_dim:
                        value = np.copy(Z[i, indices[l], n])
                        Z[i, indices[l], n] = 0
                        Z[i, np.random.randint(0, z_dim - 1), n] = value
                indices = np.where(Z[i, :, n] != 0)[0]
                diff_indices = np.diff(indices)
                condition = np.sum(np.where(diff_indices <= dictionary_dim)[0] + 1)

    # you can set the convolutional dictionaries here
    H = np.flip(HT, axis=0)
    data_generator.get_layer("decoder").set_weights(
        [H.reshape(dictionary_dim, num_conv, 1)]
    )

    # generate data now
    y = data_generator.predict(Z)

    # divide data into train and test sets
    y_train = y[0:num_train, :, 0].reshape(num_train, input_dim, 1)
    y_test = y[num_train:, :, 0].reshape(num_test, input_dim, 1)
    Z_train = Z[0:num_train, :, :]
    Z_test = Z[num_train:, :, :]

    return y_train, y_test, Z_train, Z_test


def generateData2D(
    HT, input_dim, dictionary_dim, num_conv, num_train, num_test, mean_std, density
):
    z_dim = [input_dim[0] - dictionary_dim[0] + 1, input_dim[1] - dictionary_dim[1] + 1]
    num_data = num_train + num_test

    # input
    input_signal = Input(
        shape=(z_dim[0], z_dim[1], num_conv), name="input"
    )  # Input placeholder
    # Zero-pad
    input_signal_padded = ZeroPadding2D(
        padding=(
            (dictionary_dim[0] - 1, dictionary_dim[0] - 1),
            (dictionary_dim[1] - 1, dictionary_dim[1] - 1),
        ),
        name="zeropad",
    )(input_signal)
    # build convolution
    decoded = Conv2D(
        filters=1,
        kernel_size=dictionary_dim,
        padding="valid",
        use_bias=False,
        activation=None,
        trainable=False,
        input_shape=(z_dim[0], z_dim[1], num_conv),
        name="decoder",
    )(input_signal_padded)

    # output Y = HZ
    data_generator = Model(input_signal, decoded)

    # generate sparse codes
    Z = np.zeros((num_data, z_dim[0], z_dim[1], num_conv))
    for i in range(num_data):
        for n in range(num_conv):
            # rvs = stats.uniform(amp_range[0],amp_range[1]-amp_range[0]).rvs
            rvs = stats.norm(loc=mean_std[n, 0], scale=mean_std[n, 1]).rvs
            temp = sc.sparse.random(z_dim[0], z_dim[1], density, data_rvs=rvs).A
            sign = np.concatenate(
                (
                    np.ones((temp.shape[0], temp.shape[1], 1)),
                    -np.ones((temp.shape[0], temp.shape[1], 1)),
                ),
                axis=0,
            )
            np.random.shuffle(sign)
            Z[i, :, :, n] = temp * sign[0 : temp.shape[0], 0 : temp.shape[1], 0]

    # you can set the convolutional dictionaries here
    H = np.flip(np.flip(HT, axis=0), axis=1)
    data_generator.get_layer("decoder").set_weights(
        [H.reshape(dictionary_dim[0], dictionary_dim[1], num_conv, 1)]
    )

    # generate data now
    y = data_generator.predict(Z)

    # divide data into train and test sets
    y_train = y[0:num_train, :, :, 0].reshape(num_train, input_dim[0], input_dim[1], 1)
    y_test = y[num_train:, :, :, 0].reshape(num_test, input_dim[0], input_dim[1], 1)
    Z_train = Z[0:num_train, :, :, :]
    Z_test = Z[num_train:, :, :, :]

    return y_train, y_test, Z_train, Z_test


def generate_simulated_data1d(
    input_dim,
    dictionary_dim,
    num_conv,
    num_train,
    num_test,
    snr,
    folder_name,
    distribution="norm",
    sampling_rate=30000,
    firing_rate_range=[20, 40],
    amp_range=[150, 170],
    std_range=[20, 40],
):
    num_data = num_train + num_test

    firing_rate = []
    num_spikes_per_example = []
    density = []
    for n in range(num_conv):
        firing_rate.append(
            np.random.uniform(firing_rate_range[0], firing_rate_range[1])
        )
        num_spikes_per_example.append(
            np.int(np.ceil((input_dim / sampling_rate) * firing_rate[-1]))
        )
        density.append((num_spikes_per_example[-1] + 1) / input_dim)
    print("firing_rate:", np.round(firing_rate, 4))
    print("number of spikes per example:", num_spikes_per_example)
    print("density used for code generation:", np.round(density, 4))

    print("snr:", snr)

    h = np.load("../experiments/{}/data/H_true.npy".format(folder_name))
    max_value = np.max(abs(h))

    # amplitude range of codes Z
    max_value = np.max(abs(h))
    mean_std = np.zeros((num_conv, 2))

    for n in range(num_conv):
        mean_std[n, 0] = np.random.uniform(amp_range[0], amp_range[1]) / max_value
        mean_std[n, 1] = np.random.uniform(std_range[0], std_range[1])

    y_train, y_test, z_train, z_test = generateDataNonOverlapping(
        h, input_dim, dictionary_dim, num_conv, num_train, num_test, mean_std, density
    )

    hf = h5py.File("../experiments/{}/data/data_new.h5".format(folder_name), "w")

    # calculate std from snr
    noiseSTD_all = []
    indices = np.where(y_train != 0)
    y_std = np.std(y_train[indices])
    print("y_std:", y_std)

    noiseSTD = y_std / np.power(10, (snr / 20))
    print("noiseSTD:", noiseSTD)
    noise = noiseSTD * np.random.randn(num_data, input_dim, 1)
    y_train_noisy = y_train + noise[:num_train, :, :]
    y_test_noisy = y_test + noise[num_train:, :, :]

    # NORMALIZED ALL DATA
    max_y = np.max([np.max(np.abs(y_train_noisy)), np.max(np.abs(y_test_noisy))])
    y_train /= max_y
    y_train_noisy /= max_y
    y_test /= max_y
    y_test_noisy /= max_y
    z_train /= max_y
    z_test /= max_y
    noiseSTD /= max_y
    print("max_y:", max_y)
    print("new noiseSTD:", noiseSTD)

    # # STANDARDIZED ALL DATA
    # y_train_std = np.std(y_train_noisy,axis=1)[0]
    # y_test_std = np.std(y_test_noisy,axis=1)[0]
    # y_train_noisy /= y_train_std
    # y_test_noisy /= y_test_std
    # y_train /= y_train_std
    # y_test /= y_test_std
    # for n in range(num_conv):
    #     z_train[:,:,n] /= y_train_std
    #     z_test[:,:,n] /= y_test_std

    # y_test /= noiseSTD
    # y_train /= noiseSTD
    # y_train_noisy /= noiseSTD
    # y_test_noisy /= noiseSTD
    # z_train /= noiseSTD
    # z_test /= noiseSTD
    # noiseSTD /= noiseSTD
    # #
    # print(np.max(y_train_noisy))
    # print(np.max(y_test_noisy))

    hf.create_dataset("max_y", data=max_y)
    hf.create_dataset("num_data", data=num_data)
    hf.create_dataset("num_train", data=num_train)
    hf.create_dataset("num_test", data=num_test)
    hf.create_dataset("input_dim", data=input_dim)
    hf.create_dataset("sampling_rate", data=sampling_rate)
    hf.create_dataset("firing_rate", data=firing_rate)
    hf.create_dataset("num_spikes_per_example", data=num_spikes_per_example)
    hf.create_dataset("density", data=density)
    hf.create_dataset("mean_std", data=mean_std)
    hf.create_dataset("snr", data=snr)
    hf.create_dataset("noiseSTD", data=noiseSTD)
    hf.create_dataset("y_train", data=y_train, compression="gzip", compression_opts=9)
    hf.create_dataset("y_test", data=y_test, compression="gzip", compression_opts=9)
    hf.create_dataset("z_train", data=z_train, compression="gzip", compression_opts=9)
    hf.create_dataset("z_test", data=z_test, compression="gzip", compression_opts=9)
    hf.create_dataset(
        "y_train_noisy", data=y_train_noisy, compression="gzip", compression_opts=9
    )
    hf.create_dataset(
        "y_test_noisy", data=y_test_noisy, compression="gzip", compression_opts=9
    )

    hf.close()


@make_dataset.command()
@click.option("--folder_name", default="", help="folder name in experiment directory")
def generate_simulated_data1d_from_folder(folder_name):
    # np.random.seed(85)
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
    # generate and save data
    generate_simulated_data1d(
        config_m["input_dim"],
        config_m["dictionary_dim"],
        config_m["num_conv"],
        config_d["num_train"],
        config_d["num_test"],
        config_d["snr"],
        folder_name,
        distribution=config_d["distribution"],
        sampling_rate=config_d["sampling_rate"],
        firing_rate_range=config_d["firing_rate_range"],
        amp_range=config_d["amp_range"],
        std_range=config_d["std_range"],
    )


if __name__ == "__main__":
    make_dataset()
