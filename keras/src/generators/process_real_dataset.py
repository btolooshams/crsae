"""
Copyright (c) 2019 CRISP

functions for process real data

:author: Bahareh Tolooshams
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import h5py
import click

sys.path.append("..")
PATH = sys.path[-1]


@click.group(chain=True)
def process_real_dataset():
    pass


@process_real_dataset.command()
@click.option("--input_dim", default=3000, help="dimension of each window.")
def process_neural_data_filtered_d533101(input_dim):
    """
    process data to make it ready for training purposes
    """
    num_channels = 4
    noiseSTD = [24.4, 13.5, 12.8, 13]

    # read data
    filename = "{}/data/raw/real_data/filtered_d533101.mat".format(PATH)
    data = sio.loadmat(filename)
    signal = data["filt_sig"]

    y_train_all_channels = []
    y_test_all_channels = []
    y_series_all_channels = []

    hf = h5py.File("../data/processed/real_data_{}.h5".format(input_dim), "w")

    for ch in range(num_channels):
        if ch == 0:
            # get data from ch
            signal_ch = signal[ch, :]

            # remove burst from data
            y1 = np.copy(signal_ch[:1095000])
            y2 = np.copy(signal_ch[1103000:])
            temp = np.concatenate([y1, y2])
            length_of_data = np.int(np.floor(temp.shape[0]))
            num_data = np.int(np.floor(length_of_data / input_dim))
            print("ch:", ch)
            print("number of data:", num_data)
            print("length of each example:", input_dim)
            y_series = np.copy(temp)

            # print estimated std of noise from median of y_series
            print(
                "noise std estimate for ch %i:" % ch,
                np.median(abs(y_series)) / (4 * 0.6745),
            )

            temp = temp[: num_data * input_dim]

            # change data into windows of size input_dim
            y = temp.reshape(num_data, input_dim)

            num_train = np.int(np.floor(0.97 * num_data))
            num_test = num_data - num_train

            print("num_data:", num_data)
            print("num_train:", num_train)
            print("num_test:", num_test)

            # shuffle data to pick train and test
            indices = np.arange(0, num_data, 1)
            np.random.shuffle(indices)
            y_train = np.expand_dims(y[indices[:num_train], :], axis=2)
            y_test = np.expand_dims(y[indices[num_train:], :], axis=2)
            y_series = np.expand_dims(np.expand_dims(y_series, axis=0), axis=2)

            # normalize data
            max_y = np.max(np.abs(y_series))
            y_series /= max_y
            y_train /= max_y
            y_test /= max_y
            noiseSTD /= max_y

            g = hf.create_group("{}".format(ch))
            g.create_dataset(
                "y_series", data=y_series, compression="gzip", compression_opts=9
            )
            g.create_dataset(
                "y_train", data=y_train, compression="gzip", compression_opts=9
            )
            g.create_dataset(
                "y_test", data=y_test, compression="gzip", compression_opts=9
            )
            g.create_dataset("num_data", data=num_data)
            g.create_dataset("num_train", data=num_train)
            g.create_dataset("num_test", data=num_test)
            g.create_dataset("input_dim", data=input_dim)
            g.create_dataset("length_of_data", data=length_of_data)
            g.create_dataset("noiseSTD", data=noiseSTD[ch])
            g.create_dataset("max_y", data=max_y)

    hf.close()


@process_real_dataset.command()
@click.option("--threshold", default=1800, help="threshold to identify spike.")
@click.option(
    "--event_range",
    default=20,
    help="range of event (related to length of action potentials).",
)
def extract_spikes_filtered_d533101(threshold=1800, event_range=20):
    """
    extract spikes
    """
    filename = "../data/raw/real_data/d533101_intracellular.mat"
    data = sio.loadmat(filename)
    intracellular = data["intracellular"]
    ch = 0

    # remove burst
    y1 = np.copy(intracellular[ch, :1095000])
    y2 = np.copy(intracellular[ch, 1103000:])
    signal = np.concatenate([y1, y2])
    length_of_data = signal.shape[0]
    print("ch:", ch)
    print("length of data:", length_of_data)

    spikes = np.copy(signal)

    # count as spike if signal is above threshold
    spikes[np.where(abs(signal) >= threshold)] = 1
    spikes[np.where(abs(signal) < threshold)] = 0

    # count as only one spike, for values of 1 within the event_range
    for i in range(len(spikes)):
        if spikes[i] == 0:
            continue
        spikes[i + 1 : i + event_range] *= 0

    hf = h5py.File("../data/processed/spikes.h5", "w")
    g = hf.create_group("{}".format(ch))
    g.create_dataset("spikes", data=spikes, compression="gzip", compression_opts=9)
    hf.close()


@process_real_dataset.command()
@click.option(
    "--filename",
    default="../data/raw/real_data/filtered_d533101.mat",
    help="path to data.",
)
@click.option("--waveform_length", default=45, help="legnth of action potential.")
@click.option(
    "--numOfelements", default=5, help="number of distinct filters to extract."
)
def generateMinCorrTemplates(
    filename="../data/raw/real_data/filtered_d533101.mat",
    waveform_length=45,
    numOfelements=5,
):
    """
    find 'numOfelements' minimum correlation waveforms from neural data
    """
    data = sio.loadmat(filename)
    signal = data["filt_sig"]

    # Compute background noise level
    channel = 0
    startidx = int(6.75e5)
    endidx = int(7e5)

    # error_norm = np.linalg.norm(signal[0,startidx:endidx])/np.sqrt(endidx-startidx)
    # print(error_norm)

    # Conventional Threshold
    # thresh = 4 * np.median(abs(signal[channel,:]))/0.6745 #45 very low!!

    channel = 0
    thresh = -200
    startidx = 0

    left = 10
    right = waveform_length - left

    indices = np.where(signal[channel, startidx:] < thresh)[0]
    print(np.shape(indices))

    waveforms = np.zeros((np.size(indices), waveform_length))

    for i in np.arange(np.size(indices)):
        idx = indices[i]
        waveforms[i, :] = signal[channel, idx - left : idx + right]

    # Form a dictionary that is least correlated
    numOfwaveforms = np.shape(waveforms)[0]
    print(numOfwaveforms)
    min_corr = (np.power(numOfelements, 2) - numOfelements) / 2
    min_indices = np.zeros(numOfelements)

    for i in np.arange(5000):
        idx = np.random.choice(numOfwaveforms, numOfelements, replace=0)
        temp = waveforms[idx, :]

        corr = np.triu(np.corrcoef(temp), k=1)
        corr = np.sum(abs(corr))
        if corr < min_corr:
            min_corr = corr
            min_indices = idx

    print(min_corr)

    h = waveforms[min_indices, :].T
    # normalize filters
    h /= np.linalg.norm(h, axis=0)

    return h


if __name__ == "__main__":
    process_real_dataset()
