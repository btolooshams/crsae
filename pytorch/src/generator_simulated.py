"""
Copyright (c) 2020 Bahareh Tolooshams

data generator

:author: Bahareh Tolooshams
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
import os
import h5py


def sample_var(dataset, real_H):
    return np.dot(dataset.samples.T, real_H.t()).var(1).mean()


class SimulatedDataset1D(Dataset):
    def __init__(self, path, hyp, transform=None):

        hf = h5py.File(os.path.join(path, "data.h5"), "r")
        y = np.array(hf.get("y_train_noisy"))
        z = np.array(hf.get("z_train"))
        mu = np.array(hf.get("mu_train"))
        rate = np.array(hf.get("rate_train"))

        self.num_data = y.shape[0]
        self.num_groups = hyp["num_groups"]
        self.max_num_groups = hyp["max_num_groups"]
        self.num_trials = hyp["num_trials"]
        self.device = hyp["device"]

        self.max_num_trials = np.int(self.num_data / self.max_num_groups)

        y_group = np.zeros((self.num_groups, y.shape[1], y.shape[2]))
        mu_group = np.zeros((self.num_groups, mu.shape[1], mu.shape[2]))
        z_group = np.zeros((self.num_groups, z.shape[1], z.shape[2]))
        rate_group = np.zeros((self.num_groups, rate.shape[1], 1))

        for g in range(self.num_groups):
            start_idx = self.max_num_trials * g
            end_idx = start_idx + self.num_trials

            y_group[g, :, :] = np.mean(y[start_idx:end_idx, :, :], axis=0)
            mu_group[g, :, :] = np.mean(mu[start_idx:end_idx, :, :], axis=0)
            z_group[g, :, :] = np.mean(z[start_idx:end_idx, :, :], axis=0)
            rate_group[g, :, 0] = np.mean(rate[start_idx:end_idx, :], axis=0)

        self.y = y_group
        self.z = np.swapaxes(z_group, -1, -2)
        self.mu = mu_group
        self.rate = rate_group
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        y = self.y[idx]
        z = self.z[idx]
        mu = self.mu[idx]
        rate = self.rate[idx]

        if self.transform:
            y = torch.squeeze(self.transform(y).float(), dim=-1)
            z = self.transform(z).float()
            mu = torch.squeeze(self.transform(mu).float(), dim=-1)
            rate = torch.squeeze(self.transform(rate).float(), dim=-1)
        return y, z, mu, rate


def get_simulated1d_loader(batch_size, path, hyp, shuffle=False):
    loader = DataLoader(
        SimulatedDataset1D(
            path,
            hyp,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        shuffle=shuffle,
        batch_size=batch_size,
    )

    return loader


def generate_sparse_samples_1d(n, dim, ones, unif=True):
    samples = np.zeros((n, dim))
    for i in range(n):
        ind = np.random.choice(dim, ones, replace=False)
        if unif:
            # draws amplitude from [-5,-4] U [4,5] uniformly
            samples[i][ind] = np.random.uniform(4, 5, ones) * (
                (np.random.uniform(0, 1, ones) > 0.5) * 2 - 1
            )
        else:
            # amplitude is 1 or -1 .5 prob of each
            samples[i][ind] = np.array([1] * ones) * (
                (np.random.uniform(0, 1, ones) > 0.5) * 2 - 1
            )
    return samples.T


def generate_sparse_samples_2d(n, dim, ones, unif=True):
    samples = np.zeros((n, dim, dim))
    for i in range(n):
        ind = np.random.choice(dim, ones, replace=False)
        if unif:
            # draws amplitude from [-5,-4] U [4,5] uniformly
            temp = np.random.uniform(4, 5, ones) * (
                (np.random.uniform(0, 1, ones) > 0.5) * 2 - 1
            )
            for k in range(len(ind)):
                samples[i, ind[0], ind[1][k]] = temp[:, k]

        else:
            # amplitude is 1 or -1 .5 prob of each
            temp = np.array([1] * ones) * (
                (np.random.uniform(0, 1, ones) > 0.5) * 2 - 1
            )
            for k in range(len(ind)):
                samples[i, ind[0], ind[1][k]] = temp[:, k]
    return samples.T


class SparseVectorDataset1D(Dataset):
    def __init__(self, n, dim, num_conv, ones, transform=None, seed=None):
        self.num_conv = num_conv
        self.samples = []
        np.random.seed(seed)
        for conv in range(num_conv):
            self.samples.append(
                np.expand_dims(generate_sparse_samples_1d(n, dim, ones), axis=0)
            )

        self.samples = np.concatenate(self.samples, axis=0)
        self.transform = transform

    def __len__(self):
        return self.samples.shape[-1]

    def __getitem__(self, idx):
        sample = self.samples[:, :, idx].reshape(self.num_conv, -1, 1)

        if self.transform:
            sample = self.transform(sample).float()
        return sample


class SparseVectorDataset2D(Dataset):
    def __init__(self, n, dim, num_conv, ones, transform=None, seed=None):
        self.num_conv = num_conv
        self.dim = dim
        self.samples = []
        np.random.seed(seed)
        for conv in range(num_conv):
            self.samples.append(
                np.expand_dims(generate_sparse_samples_2d(n, dim, ones), axis=0)
            )

        self.samples = np.concatenate(self.samples, axis=0)
        self.transform = transform

    def __len__(self):
        return self.samples.shape[-1]

    def __getitem__(self, idx):
        sample = self.samples[:, :, :, idx].reshape(
            self.num_conv, self.dim, self.dim, 1
        )
        if self.transform:
            sample = torch.tensor(sample).float()
        return sample


class EncodingDataset(Dataset):
    def __init__(self, data_loader, net, device=None, transform=None, seed=None):
        self.samples = []
        self.c = []
        print("create encoding dataset.")
        for idx, (img, c) in tqdm(enumerate(data_loader)):
            img = img.to(device)
            output = net(img)

            self.samples.append(net.last_encoding)
            self.c.append(c)
        self.samples = torch.cat(self.samples)
        self.c = torch.cat(self.c)
        self.D_enc = net.last_encoding.shape[-1]
        self.num_conv = net.num_conv
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample).float()

        return sample, self.c[idx]