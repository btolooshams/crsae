"""
Copyright (c) 2020 CRISP

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class RELUTwosided(torch.nn.Module):
    def __init__(self, num_conv, lam=1e-3, L=100, sigma=1, device=None):
        super(RELUTwosided, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(
            lam * torch.ones(1, num_conv, 1, 1, device=device)
        )
        self.sigma = sigma
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        la = self.lam * (self.sigma ** 2)
        out = self.relu(torch.abs(x) - la / self.L) * torch.sign(x)
        return out


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        if H is None:
            H = torch.randn((self.num_conv, 1, self.dictionary_dim), device=self.device)
            H = F.normalize(H, p=2, dim=-1)
        self.register_parameter("H", torch.nn.Parameter(H))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def forward(self, x):
        num_batches = x.shape[0]

        D_in = x.shape[2]
        D_enc = F.conv1d(x, self.get_param("H"), stride=self.stride).shape[-1]

        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc))
        )

        x_old = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        yk = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        x_new = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()
        for t in range(self.T):
            H_wt = x - F.conv_transpose1d(yk, self.get_param("H"), stride=self.stride)
            x_new = (
                yk + F.conv2d(H_wt, self.get_param("H"), stride=self.stride) / self.L
            )
            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = F.conv_transpose1d(x_new, self.get_param("H"), stride=self.stride)

        return z, x_new, self.lam


class CRsAE2D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )

            if self.twosided:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                ) + (x_new < -(self.lam / self.L)).float() * (
                    x_new + (self.lam / self.L)
                )
            else:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                )

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.lam


class CRsAE2DFreeBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DFreeBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        self.b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device)
            + (hyp["lam"] / hyp["L"])
        )

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )

            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.b) * torch.sign(x_new)
            else:
                x_new = self.relu(x_new - self.b)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.b


class CRsAE2DUntied(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DUntied, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))

        We = torch.clone(H)
        Wd = torch.clone(H)

        self.register_parameter("H", torch.nn.Parameter(H))
        self.register_parameter("We", torch.nn.Parameter(We))
        self.register_parameter("Wd", torch.nn.Parameter(Wd))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )
        self.get_param("We").data = F.normalize(
            self.get_param("We").data, p="fro", dim=(-1, -2)
        )
        self.get_param("Wd").data = F.normalize(
            self.get_param("Wd").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk
                + F.conv2d(x_tilda, self.get_param("We"), stride=self.stride) / self.L
            )

            if self.twosided:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                ) + (x_new < -(self.lam / self.L)).float() * (
                    x_new + (self.lam / self.L)
                )
            else:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                )

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("Wd"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.lam


class CRsAE2DUntiedFreeBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DUntiedFreeBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))

        We = torch.clone(H)
        Wd = torch.clone(H)

        self.register_parameter("H", torch.nn.Parameter(H))
        self.register_parameter("We", torch.nn.Parameter(We))
        self.register_parameter("Wd", torch.nn.Parameter(Wd))

        self.b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device)
            + (hyp["lam"] / hyp["L"])
        )

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )
        self.get_param("We").data = F.normalize(
            self.get_param("We").data, p="fro", dim=(-1, -2)
        )
        self.get_param("Wd").data = F.normalize(
            self.get_param("Wd").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk
                + F.conv2d(x_tilda, self.get_param("We"), stride=self.stride) / self.L
            )

            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.b) * torch.sign(x_new)
            else:
                x_new = self.relu(x_new - self.b)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("Wd"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.b


class CRsAE2DTrainableBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DTrainableBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.lam = hyp["lam"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.sigma = hyp["sigma"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        self.relu = RELUTwosided(
            self.num_conv, self.lam, self.L, self.sigma, self.device
        )

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        yk = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.relu.lam


class CRsAE2DUntiedTrainableBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DUntiedTrainableBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]
        self.sigma = hyp["sigma"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))

        We = torch.clone(H)
        Wd = torch.clone(H)

        self.register_parameter("H", torch.nn.Parameter(H))
        self.register_parameter("We", torch.nn.Parameter(We))
        self.register_parameter("Wd", torch.nn.Parameter(Wd))

        self.relu = RELUTwosided(
            self.num_conv, self.lam, self.L, self.sigma, self.device
        )

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )
        self.get_param("We").data = F.normalize(
            self.get_param("We").data, p="fro", dim=(-1, -2)
        )
        self.get_param("Wd").data = F.normalize(
            self.get_param("Wd").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("We"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk
                + F.conv2d(x_tilda, self.get_param("We"), stride=self.stride) / self.L
            )

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("Wd"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.relu.lam
