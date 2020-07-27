"""
Copyright (c) 2020 CRISP

data generator

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import gc
import utils


def train_ae(
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
):

    info_period = hyp["info_period"]
    noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    network = hyp["network"]
    supervised = hyp["supervised"]

    if normalize:
        net.normalize()

    if hyp["denoising"]:
        if test_loader is not None:

            with torch.no_grad():
                psnr = []
                for idx_test, (img_test, _) in tqdm(enumerate(test_loader)):
                    img_test_noisy = (
                        img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                    ).to(device)

                    img_test_hat, _, _ = net(img_test_noisy)

                    img_test_noisy.detach()

                    psnr.append(
                        utils.PSNR(
                            img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                            img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                        )
                    )
                np.save(os.path.join(PATH, "psnr_init.npy"), np.array(psnr))
                print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            if supervised:
                img = img.to(device)
                noise = noiseSTD / 255 * torch.randn(img.shape).to(device)

                noisy_img = img + noise

                img_hat, _, _ = net(noisy_img)

                loss = criterion(img, img_hat)
            else:
                noisy_img = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)

                img_hat, _, _ = net(noisy_img)

                loss = criterion(noisy_img, img_hat)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            if idx % info_period == 0:
                print("loss:{:.8f} ".format(loss.item()))

            torch.cuda.empty_cache()

        # ===================log========================

        if hyp["denoising"]:
            if test_loader is not None:
                with torch.no_grad():
                    psnr = []
                    for idx_test, (img_test, _) in tqdm(enumerate(test_loader)):
                        img_test_noisy = (
                            img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                        ).to(device)

                        img_test_hat, _, _ = net(img_test_noisy)

                        psnr.append(
                            utils.PSNR(
                                img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                                img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                            )
                        )
                    np.save(
                        os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)),
                        np.array(psnr),
                    )
                    print("")
                    print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.8f} ".format(
                epoch + 1, hyp["num_epochs"], loss.item()
            )
        )

    return net


def train_ae_withtrainablebias(
    net,
    data_loader,
    hyp,
    criterion_ae,
    criterion_lam,
    optimizer_ae,
    optimizer_lam,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
):

    info_period = hyp["info_period"]

    net.normalize()

    noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    network = hyp["network"]
    supervised = hyp["supervised"]

    if hyp["denoising"]:
        if test_loader is not None:
            with torch.no_grad():
                psnr = []
                for idx_test, (img_test, _) in tqdm(enumerate(test_loader)):
                    img_test_noisy = (
                        img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                    ).to(device)

                    img_test_hat, _, _ = net(img_test_noisy)

                    psnr.append(
                        utils.PSNR(
                            img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                            img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                        )
                    )
                np.save(os.path.join(PATH, "psnr_init.npy"), np.array(psnr))
                print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader)):
            optimizer_ae.zero_grad()
            optimizer_lam.zero_grad()
            if supervised:
                img = img.to(device)
                noise = noiseSTD / 255 * torch.randn(img.shape).to(device)

                # ===================forward=====================
                output = net(img + noise)

                loss_ae = criterion_ae(img, output[0])
                loss_lam = criterion_lam(output[1:], hyp)
            else:
                noisy_img = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)
                # ===================forward=====================
                output = net(noisy_img)

                loss_ae = criterion_ae(noisy_img, output[0])
                loss_lam = criterion_lam(output[1:], hyp)

            loss_all += loss_ae.item()
            # ===================backward====================
            optimizer_ae.zero_grad()
            optimizer_lam.zero_grad()

            loss_ae.backward(retain_graph=True)
            optimizer_ae.step()

            loss_lam.backward()
            optimizer_lam.step()

            if normalize:
                net.normalize()

            if idx % info_period == 0:
                print("loss ae:{:.8f} ".format(loss_ae.item()))

        # ===================log========================

        if hyp["denoising"]:
            if test_loader is not None:
                with torch.no_grad():
                    psnr = []
                    for idx_test, (img_test, _) in tqdm(enumerate(test_loader)):
                        img_test_noisy = (
                            img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                        ).to(device)

                        img_test_hat, _, _ = net(img_test_noisy)

                        psnr.append(
                            utils.PSNR(
                                img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                                img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                            )
                        )
                    np.save(
                        os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)),
                        np.array(psnr),
                    )
                    print("PSNR: {}".format(np.round(np.array(psnr), decimals=4)))

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.8f} ".format(
                epoch + 1, hyp["num_epochs"], loss_ae.item()
            )
        )

    return net
