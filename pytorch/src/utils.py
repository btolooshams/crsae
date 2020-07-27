"""
Copyright (c) 2020 CRISP

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import MS_SSIM


class MSSSIM_l1(torch.nn.Module):
    def __init__(self):
        super(MSSSIM_l1, self).__init__()

        self.a = 0.8
        self.win_size = 4
        self.data_range = 255
        self.channel = 1

    def forward(self, input, target):

        loss = self.a * (
            1
            - MS_SSIM(
                win_size=self.win_size, data_range=self.data_range, channel=self.channel
            )(input, target)
        ) + (1 - self.a) * torch.nn.L1Loss()(input, target)
        return loss


class LambdaLoss2D(torch.nn.Module):
    def __init__(self):
        super(LambdaLoss2D, self).__init__()

    def forward(self, target, hyp):
        enc_hat = target[0]
        lam = target[1]

        num_conv = hyp["num_conv"]
        delta = hyp["delta"]
        lam_init = hyp["lam"]

        r = delta * lam_init

        Ne = enc_hat.shape[-1] * enc_hat.shape[-2]

        lam_enc_l1 = torch.sum(torch.abs(lam * enc_hat), dim=(-1, -2, -3))
        return torch.mean(
            lam_enc_l1
            + torch.sum(lam * delta, dim=(-1, -2, -3))
            - (Ne + r - 1) * torch.sum(torch.log(lam), dim=(-1, -2, -3))
        )


def normalize1d(x):
    return F.normalize(x, dim=-1)


def normalize2d(x):
    return F.normalize(x, dim=(-1, -2))


def err1d_H(H, H_hat):
    H = H.detach().clone().cpu().numpy()
    H_hat = H_hat.detach().clone().cpu().numpy()

    H /= np.linalg.norm(H, axis=-1, keepdims=True)
    H_hat /= np.linalg.norm(H_hat, axis=-1, keepdims=True)

    num_conv = H.shape[0]

    err = []
    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :] * H_hat[conv, 0, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def err2d_H(H, H_hat):
    H = H.detach().clone().cpu().numpy()
    H_hat = H_hat.detach().clone().cpu().numpy()

    num_conv = H.shape[0]

    err = []

    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :, :] * H_hat[conv, 0, :, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def PSNR(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    max_x = np.max(x)
    return 20 * np.log10(max_x) - 10 * np.log10(mse)


def calc_pad_sizes(x, dictionary_dim=8, stride=1):
    left_pad = stride
    right_pad = (
        0
        if (x.shape[3] + left_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[3] + left_pad - dictionary_dim) % stride)
    )
    top_pad = stride
    bot_pad = (
        0
        if (x.shape[2] + top_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[2] + top_pad - dictionary_dim) % stride)
    )
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad
