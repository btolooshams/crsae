"""
Copyright (c) 2020 Bahareh Tolooshams

data generator

:author: Bahareh Tolooshams
"""

import torch
from tqdm import tqdm
import os
import numpy as np
import gc
import utils


def train_simulated_ae_1d(
    net, data_loader, num_epochs, criterion, optimizer, real_H, device=None
):
    err = []

    net.normalize()

    err.append(utils.err1d_H(real_H, net.H))
    print("initial, err_H:{}".format(err[-1]))

    for epoch in tqdm(range(num_epochs)):
        for data in tqdm(data_loader):

            img = real_H(data.view(-1, net.num_conv, net.D_enc))

            img = img.to(device)

            # ===================forward=====================
            output = net(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.zero_mean()
            net.normalize()

        err.append(utils.err1d_H(real_H, net.H))
        # ===================log========================
        print(
            "epoch [{}/{}], loss:{:.4f}, err_H:{}".format(
                epoch + 1, num_epochs, loss.data, err[-1]
            )
        )
    return net, err


def train_simulated_ae_2d(
    net, data_loader, num_epochs, criterion, optimizer, real_H, device=None
):
    err = []

    net.normalize()

    err.append(utils.err2d_H(real_H, net.H))
    print("initial, err_H:{}".format(err[-1]))

    for epoch in tqdm(range(num_epochs)):
        for data in tqdm(data_loader):

            img = real_H(data.view(-1, net.num_conv, net.D_enc, net.D_enc))

            img = img.to(device)

            # ===================forward=====================
            output = net(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.zero_mean()
            net.normalize()

        err.append(utils.err2d_H(real_H, net.H))
        # ===================log========================
        print(
            "epoch [{}/{}], loss:{:.4f}, err_H:{}".format(
                epoch + 1, num_epochs, loss.data, err[-1]
            )
        )
    return net, err

def train_ae_simulated(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
    H_true=None,
):
    err = []
    min_err = None

    info_period = hyp["info_period"]
    noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    network = hyp["network"]

    if normalize:
        net.normalize()

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (y, _, mu, _) in tqdm(enumerate(data_loader)):

            y_zero_mean = y - torch.nn.Sigmoid()(mu)
            y_zero_mean = y_zero_mean.to(device)

            # ===================forward=====================
            y_hat, _, _ = net(y_zero_mean)

            loss = criterion(y_zero_mean, y_hat)

            loss_all += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            if idx % info_period == 0:
                if H_true is not None:
                    err_H = utils.err1d_H(H_true, net.H.weight)
                    print(
                        "loss:{:.4f}, err_H:{:4f}\n".format(loss.item(), np.mean(err_H))
                    )
                else:
                    print("loss:{:.4f}\n".format(loss.item()))

        # ===================log========================

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

        if network != "LCSC2D":
            torch.save(
                net.H.weight.data, os.path.join(PATH, "H_epoch{}.pt".format(epoch))
            )
            if H_true is not None:
                torch.save(
                    utils.err1d_H(H_true, net.H.weight),
                    os.path.join(PATH, "err_H_epoch{}.pt".format(epoch)),
                )

        err.append(loss.item())
        if H_true is None:
            print(
                "epoch [{}/{}], loss:{:.4f} ".format(
                    epoch + 1, hyp["num_epochs"], loss.item()
                )
            )
        else:
            print(
                "epoch [{}/{}], loss:{:.4f}, err_H:{:4f}".format(
                    epoch + 1, hyp["num_epochs"], loss.item(), np.mean(err_H)
                )
            )

    return net
