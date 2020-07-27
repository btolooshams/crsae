"""
Copyright (c) 2019 CRISP

Plot helpers.

:author: Bahareh Tolooshams
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import itertools
import time
from time import gmtime, strftime
import sys


sys.path.append("..")
PATH = sys.path[-1]

from src.plotter.plot_helpers import *


def plot_lr_loss(
    PATH,
    folder_name,
    line_width=2,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    hf = h5py.File(
        "{}/experiments/{}/results/results_lr.h5".format(PATH, folder_name), "r"
    )
    iterations = np.array(hf.get("iterations"))
    lr = np.array(hf.get("lr"))
    loss_lr = np.array(hf.get("loss_lr"))
    hf.close()

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    plt.plot(lr, loss_lr, lw=line_width, color="black")
    plt.ylabel("$\mathrm{loss}$", fontweight="bold")
    plt.xlabel("$\mathrm{lr}$", fontweight="bold")
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/{}/reports/lr.pdf".format(PATH, folder_name))

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    plt.plot(lr[:30], loss_lr[:30], lw=line_width, color="black")
    plt.ylabel("$\mathrm{loss}$", fontweight="bold")
    plt.xlabel("$\mathrm{lr}$", fontweight="bold")
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/{}/reports/lr_zoom.pdf".format(PATH, folder_name))


def plot_lr_iterations(
    lr_iterations,
    num_epochs,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # best_epoch = np.argmin(val_loss)
    x_values = np.linspace(1, num_epochs, lr_iterations.shape[0])

    x_lim = [1, num_epochs]
    y_lim = [0.9 * np.min(lr_iterations), 1.1 * np.max(lr_iterations)]

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    plt.plot(x_values, lr_iterations, lw=line_width, color="black")
    plt.ylabel("$\mathrm{lr}$", fontweight="bold")
    plt.xlabel("$\mathrm{epochs}$", fontweight="bold")
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/lr_iterations_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_loss(
    val_loss,
    train_loss,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = val_loss.shape[0]
    # best_epoch = np.argmin(val_loss)
    x_values = np.linspace(1, num_epochs, num_epochs)
    x_lim = [1, num_epochs]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min([np.min(val_loss), np.min(train_loss)]))
    y_lim[1] = np.round(1.05 * np.max([np.max(val_loss), np.max(train_loss)]))

    # plot
    k = 1
    ax = fig.add_subplot(111)
    plt.plot(x_values, val_loss, lw=line_width, color="r")
    plt.plot(
        x_values[::k], val_loss[::k], "vr", label="$\mathrm{validation}$", lw=line_width
    )
    plt.plot(x_values[best_epoch], val_loss[best_epoch], "vb", lw=line_width)

    plt.plot(x_values, train_loss, lw=line_width, color="g")
    plt.plot(
        x_values[::k], train_loss[::k], ".g", label="$\mathrm{training}$", lw=line_width
    )

    for i in range(len(best_val_epoch)):
        plt.plot(
            x_values[best_val_epoch[i]],
            train_loss[best_val_epoch[i]],
            ".k",
            lw=line_width,
        )

    plt.plot(
        x_values[best_epoch],
        train_loss[best_epoch],
        ".b",
        label="$\mathrm{learned}$",
        lw=line_width,
    )

    plt.ylabel("$\mathrm{Loss}$")
    plt.xlabel("$\mathrm{Epochs}$")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    # if (y_lim[1] - y_lim[0] != 0):
    #     plt.yticks(
    #         [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 4)]
    #     )
    #     plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/loss_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_sporco_loss(
    l2_loss,
    fit_time,
    PATH,
    folder_name,
    file_number,
    row=1,
    y_fine=0.5,
    line_width=2.2,
    marker_size=15,
    scale=4,
    scale_height=0.75,
    text_font=45,
    title_font=45,
    axes_font=48,
    legend_font=32,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_iter = l2_loss.shape[0]
    best_epoch = np.argmin(l2_loss)

    # plot
    k = 1
    ax = fig.add_subplot(121)
    x_values = np.linspace(0, fit_time, num_iter)
    plt.plot(x_values, l2_loss, lw=line_width, color="k")
    plt.plot(x_values[::k], l2_loss[::k], "vk", lw=line_width)
    plt.plot(x_values[best_epoch], l2_loss[best_epoch], "vb", lw=line_width)

    plt.ylabel("$\mathrm{l2\;loss}$")
    plt.xlabel("$\mathrm{Time\;[s]}$")
    # plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(122)
    x_values = np.linspace(1, num_iter, num_iter)
    x_lim = [1, num_iter]
    plt.plot(x_values, l2_loss, lw=line_width, color="k")
    plt.plot(x_values[::k], l2_loss[::k], "vk", lw=line_width)
    plt.plot(x_values[best_epoch], l2_loss[best_epoch], "vb", lw=line_width)

    plt.ylabel("$\mathrm{l2\;loss}$")
    plt.xlabel("$\mathrm{Iterations}$")
    plt.xticks([i for i in range(0, len(x_values) + 5, np.int(np.ceil(num_iter / 5)))])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/loss_sporco_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_loss_crsae_vs_sporco(
    crsae_loss,
    sporco_loss,
    crsae_fit_time,
    sporco_fit_time,
    PATH,
    folder_name,
    crsae_file_number,
    sporco_file_number,
    row=1,
    y_fine=0.5,
    line_width=2.2,
    marker_size=15,
    scale=4,
    scale_height=0.75,
    text_font=45,
    title_font=45,
    axes_font=48,
    legend_font=32,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    # plot
    k = 1
    ax = fig.add_subplot(121)
    num_iter = crsae_loss.shape[0]
    x_values = np.linspace(0, crsae_fit_time, num_iter)
    best_epoch = np.argmin(crsae_loss)
    best_crsae = crsae_loss[best_epoch]
    plt.plot(x_values, crsae_loss, lw=line_width, color="g")
    # plt.plot(x_values[::k], crsae_loss[::k], "og", lw=line_width)
    plt.plot(x_values[best_epoch], crsae_loss[best_epoch], "vb", markersize=marker_size)

    print("crsae:", x_values[best_epoch])

    num_iter = sporco_loss.shape[0]
    x_values = np.linspace(0, sporco_fit_time, num_iter)
    best_epoch = np.argmin(sporco_loss)
    better_epoch = np.where(sporco_loss <= best_crsae)[0][0]
    plt.plot(x_values, sporco_loss, lw=line_width, color="k")
    # plt.plot(x_values[::k], sporco_loss[::k], "vk", lw=line_width)
    # plt.plot(x_values[best_epoch], sporco_loss[best_epoch], "vb", lw=line_width)
    plt.plot(
        x_values[better_epoch], sporco_loss[better_epoch], "vb", markersize=marker_size
    )

    print("sporco:", x_values[better_epoch])

    plt.ylabel("$10\log{\|\mathbf{y} - \hat{\mathbf{y}}\|_2}$")
    plt.xlabel("$\mathrm{Time\;[s]}$")
    plt.xticks([i for i in range(0, 600 + 5, 300)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(122)
    num_iter = crsae_loss.shape[0]
    x_values = np.linspace(1, num_iter, num_iter)
    x_lim = [1, num_iter]
    best_epoch = np.argmin(crsae_loss)
    best_crsae = crsae_loss[best_epoch]
    plt.plot(x_values, crsae_loss, lw=line_width, label="CRsAE", color="g")
    # plt.plot(x_values[::k], crsae_loss[::k], "og", label="CRsAE", lw=line_width)
    plt.plot(x_values[best_epoch], crsae_loss[best_epoch], "vb", markersize=marker_size)

    print("crsae:", x_values[best_epoch])

    num_iter = sporco_loss.shape[0]
    x_values = np.linspace(1, num_iter, num_iter)
    x_lim = [1, num_iter]
    best_epoch = np.argmin(sporco_loss)
    better_epoch = np.where(sporco_loss <= best_crsae)[0][0]
    plt.plot(x_values, sporco_loss, lw=line_width, label="Sporco", color="k")
    # plt.plot(x_values[::k], sporco_loss[::k], "vk", label="Sporco", lw=line_width)
    # plt.plot(x_values[best_epoch], sporco_loss[best_epoch], "vb", lw=line_width)
    plt.plot(
        x_values[better_epoch], sporco_loss[better_epoch], "vb", markersize=marker_size
    )

    print("sporco:", x_values[better_epoch])

    # plt.ylabel("$\mathrm{l2\;loss}$")
    plt.xlabel("$\mathrm{Iterations}$")
    plt.xticks([i for i in range(0, len(x_values) + 5, np.int(np.ceil(num_iter / 2)))])
    plt.yticks([])
    plt.legend(loc="upper right", ncol=1)
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/loss_crsae_vs_sporco_{}_{}.eps".format(
            PATH, folder_name, crsae_file_number, sporco_file_number
        )
    )


def plot_H_loss(
    val_H_loss,
    train_H_loss,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = val_H_loss.shape[0]
    # best_epoch = np.argmin(val_H_loss)
    x_values = np.linspace(1, num_epochs, num_epochs)
    x_lim = [1, num_epochs]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min([np.min(val_H_loss), np.min(train_H_loss)]))
    y_lim[1] = np.round(1.05 * np.max([np.max(val_H_loss), np.max(train_H_loss)]))

    # plot
    k = 1
    ax = fig.add_subplot(111)
    plt.plot(x_values, val_H_loss, lw=line_width, color="r")
    plt.plot(
        x_values[::k],
        val_H_loss[::k],
        "vr",
        label="$\mathrm{validation}$",
        lw=line_width,
    )
    plt.plot(x_values[best_epoch], val_H_loss[best_epoch], "vb", lw=line_width)

    plt.plot(x_values, train_H_loss, lw=line_width, color="g")
    plt.plot(
        x_values[::k],
        train_H_loss[::k],
        ".g",
        label="$\mathrm{training}$",
        lw=line_width,
    )

    for i in range(len(best_val_epoch)):
        plt.plot(
            x_values[best_val_epoch[i]],
            train_H_loss[best_val_epoch[i]],
            ".k",
            lw=line_width,
        )

    plt.plot(
        x_values[best_epoch],
        train_H_loss[best_epoch],
        ".b",
        label="$\mathrm{learned}$",
        lw=line_width,
    )

    plt.ylabel("$\mathrm{mse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\mathrm{Epochs}$")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    if y_lim[1] - y_lim[0] != 0:
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 4)]
        )
        plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/loss_H_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_lambda_loss(
    val_lambda_loss,
    train_lambda_loss,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = val_lambda_loss.shape[0]
    # best_epoch = np.argmin(val_l1_norm_loss)
    x_values = np.linspace(1, num_epochs, num_epochs)
    x_lim = [1, num_epochs]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        0.95 * np.min([np.min(val_lambda_loss), np.min(train_lambda_loss)])
    )
    y_lim[1] = np.round(
        1.05 * np.max([np.max(val_lambda_loss), np.max(train_lambda_loss)])
    )

    # plot
    k = 1
    ax = fig.add_subplot(111)
    plt.plot(x_values, val_lambda_loss, lw=line_width, color="r")
    plt.plot(
        x_values[::k],
        val_lambda_loss[::k],
        "vr",
        label="$\mathrm{validation}$",
        lw=line_width,
    )
    # plt.plot(x_values[best_epoch], train_lambda_loss[best_epoch], "vb", lw=line_width)

    plt.plot(x_values, train_lambda_loss, lw=line_width, color="g")
    plt.plot(
        x_values[::k],
        train_lambda_loss[::k],
        ".g",
        label="$\mathrm{training}$",
        lw=line_width,
    )

    for i in range(len(best_val_epoch)):
        plt.plot(
            x_values[best_val_epoch[i]],
            train_lambda_loss[best_val_epoch[i]],
            ".k",
            lw=line_width,
        )

    plt.plot(
        x_values[best_epoch],
        train_lambda_loss[best_epoch],
        ".b",
        label="$\mathrm{learned}$",
        lw=line_width,
    )

    plt.ylabel("$lambda log loss$")
    plt.xlabel("$\mathrm{Epochs}$")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    # if y_lim[1] - y_lim[0] != 0:
    #     plt.yticks(
    #         [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 4)]
    #     )
    #     plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/lambda_loss_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_miss_false(
    missed_list,
    false_list,
    PATH,
    folder_name,
    file_number,
    spikes_filter,
    ch,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):
    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    x_lim = [0, np.round(np.max(missed_list))]
    y_lim = [0, np.round(np.max(false_list))]

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    if file_number == "kmeans":
        plt.plot(missed_list, false_list, ".", lw=line_width, color="black")
    else:
        plt.plot(missed_list, false_list, lw=line_width, color="black")
    plt.ylabel("$\mathrm{Estimated\;Spikes\;False}$", fontweight="bold")
    if y_lim[1] - y_lim[0] != 0:
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)]
        )
    plt.xlabel("$\mathrm{True\;Spikes\;Missed}$", fontweight="bold")
    if x_lim[1] - x_lim[0] != 0:
        plt.xticks(
            [i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )

    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/miss_false_{}_{}_{}.pdf".format(
            PATH, folder_name, file_number, spikes_filter, ch
        )
    )


def plot_crsae_cbp_miss_false(
    crsae_missed_list,
    crsae_false_list,
    cbp_missed_list,
    cbp_false_list,
    PATH,
    folder_name,
    file_number,
    spikes_filter,
    ch,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):
    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    x_lim = [0, np.round(np.max(crsae_missed_list))]
    y_lim = [0, np.round(np.max(crsae_false_list))]

    x_lim = [0, 50]
    y_lim = [0, 15]

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    plt.plot(
        crsae_missed_list, crsae_false_list, lw=line_width, color="black", label="CRsAE"
    )
    plt.plot(cbp_missed_list, cbp_false_list, lw=line_width, color="red", label="CBP")
    plt.ylabel("$\mathrm{Estimated\;Spikes\;False\;[\%]}$", fontweight="bold")
    if y_lim[1] - y_lim[0] != 0:
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)]
        )
    plt.xlabel("$\mathrm{True\;Spikes\;Missed\;[\%]}$", fontweight="bold")
    plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)])
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.legend()

    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/miss_false_{}_{}_{}.pdf".format(
            PATH, folder_name, file_number, spikes_filter, ch
        )
    )


def plot_all_miss_false(
    all_missed_list,
    all_false_list,
    PATH,
    folder_name,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=50,
    title_font=50,
    axes_font=50,
    legend_font=50,
    number_font=50,
):
    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    x_lim = [0, 50]
    y_lim = [0, 50]

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)
    ax = fig.add_subplot(111)
    for l in range(len(all_missed_list)):
        missed_list = all_missed_list[l]
        false_list = all_false_list[l]

        plt.plot(missed_list, false_list, lw=line_width, label="%s" % l)
        plt.ylabel("$\mathrm{Estimated\;Spikes\;False}$", fontweight="bold")
        plt.yticks(
            [
                i
                for i in np.arange(
                    np.floor(y_lim[0]),
                    np.ceil(y_lim[1]),
                    (np.ceil(y_lim[1]) - np.floor(y_lim[0])) / 5,
                )
            ]
        )
        plt.xlabel("$\mathrm{True\;Spikes\;Missed}$", fontweight="bold")
        plt.xticks(
            [
                i
                for i in np.arange(
                    np.floor(x_lim[0]),
                    np.ceil(x_lim[1]),
                    (np.ceil(x_lim[1]) - np.floor(x_lim[0])) / 5,
                )
            ]
        )
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    plt.legend(loc="upper right", ncol=5)
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/miss_false_all.pdf".format(PATH, folder_name)
    )


def plot_H_real(
    H_init,
    H_learned,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_init.shape[2]
    dictionary_dim = H_init.shape[0]

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.1 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(1.1 * np.min([np.min(H_init), np.min(H_learned)]), 2)
    y_lim[1] = np.round(1.5 * np.max([np.max(H_init), np.max(H_learned)]), 2)

    k = 4
    for n in range(num_conv):
        ax = fig.add_subplot(1, num_conv, n + 1)
        plt.plot(x_values, H_init[:, 0, n], lw=line_width, color="gray")
        plt.plot(
            x_values[::k],
            H_init[::k, 0, n],
            "v",
            markersize=marker_size,
            label="$\mathrm{Initial}$",
            color="gray",
        )
        plt.plot(x_values, H_learned[:, 0, n], lw=line_width, color="r")
        plt.plot(
            x_values[::k],
            H_learned[::k, 0, n],
            "*",
            markersize=marker_size,
            label="$\mathrm{Learned}$",
            color="r",
        )
        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.50)])
        plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine)])
        plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
    plt.legend(loc="upper right", ncol=1)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_H_real_2d(
    H_init,
    H_learned,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_init.shape[3]
    dictionary_dim = H_init.shape[0]

    y_lim = [0, 0]
    y_lim[0] = np.round(1.1 * np.min([np.min(H_init), np.min(H_learned)]), 2)
    y_lim[1] = np.round(1.5 * np.max([np.max(H_init), np.max(H_learned)]), 2)

    k = 4
    for n in range(num_conv):
        if num_conv <= 8:
            ax = fig.add_subplot(1, num_conv, n + 1)
        elif num_conv <= 16:
            ax = fig.add_subplot(2, 8, n + 1)
        elif num_conv <= 32:
            ax = fig.add_subplot(4, 8, n + 1)
        else:
            ax = fig.add_subplot(8, 8, n + 1)
        plt.imshow(H_learned[:, :, 0, n], cmap="gray")

        plt.xticks([])
        plt.yticks([])
        # plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
    plt.legend(loc="upper right", ncol=1)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_H_sim(
    H_true,
    H_init,
    H_learned,
    best_permutation_index,
    flip,
    delay,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.2 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(H_init), np.min(H_learned)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(H_init), np.max(H_learned)]), 2
    )
    if file_number == "2018-08-07-20-05-53":
        y_lim = [-0.5, 0.5]
        y_fine = 0.5

    k = 4
    for n in range(num_conv):
        delay_n = np.int(delay[n])
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        if n == np.int(num_conv / row) - 1:
            plt.plot(
                x_values,
                H_true[:, 0, n],
                lw=line_width * 1.2,
                label="$\mathrm{True}$",
                color="k",
            )
        else:
            plt.plot(x_values, H_true[:, 0, n], lw=line_width * 1.2, color="k")
        plt.plot(
            x_values,
            flip[n] * H_init[:, 0, permutations[best_permutation_index][n]],
            lw=line_width * 0.9,
            color="gray",
        )
        if n == np.int(num_conv / row) - 2:
            plt.plot(
                x_values[::k],
                flip[n] * H_init[::k, 0, permutations[best_permutation_index][n]],
                "v",
                markersize=marker_size,
                label="$\mathrm{Initial}$",
                color="gray",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * H_init[::k, 0, permutations[best_permutation_index][n]],
                "v",
                markersize=marker_size,
                color="gray",
            )
        plt.plot(
            x_values[:],
            flip[n] * H_learned[:, 0, permutations[best_permutation_index][n]],
            lw=line_width,
            color="r",
        )
        if n == np.int(num_conv / row) - 3:
            plt.plot(
                x_values[::k],
                flip[n] * H_learned[::k, 0, permutations[best_permutation_index][n]],
                "*",
                markersize=marker_size,
                label="$\mathrm{Learned}$",
                color="r",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * H_learned[::k, 0, permutations[best_permutation_index][n]],
                "*",
                markersize=marker_size,
                color="r",
            )

        if delay_n < 0:
            plt.plot(
                x_values[abs(delay_n) :],
                flip[n]
                * H_learned[:delay_n, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * H_learned[:delay_n:k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * H_learned[:delay_n:k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        elif delay_n > 0:
            plt.plot(
                x_values[:-delay_n],
                flip[n]
                * H_learned[delay_n:, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * H_learned[delay_n::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * H_learned[delay_n::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        if delay_n == 0:
            plt.plot(
                x_values[:],
                flip[n] * H_learned[:, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * H_learned[::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * H_learned[::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    color="g",
                )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, y_fine)])
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 2:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 3:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 4:
            plt.legend(loc="upper right", ncol=1)

        plt.ylim([y_lim[0], y_lim[1] + 0.05])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_We_sim(
    H_true,
    We_init,
    We_learned,
    best_permutation_index,
    flip,
    delay,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # normalize all for ploting
    H_true /= np.linalg.norm(H_true, axis=0)
    We_init /= np.linalg.norm(We_init, axis=0)
    We_learned /= np.linalg.norm(We_learned, axis=0)

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.2 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(We_init), np.min(We_learned)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(We_init), np.max(We_learned)]), 2
    )
    if file_number == "2018-08-07-20-05-53":
        y_lim = [-0.5, 0.5]
        y_fine = 0.5

    k = 4
    for n in range(num_conv):
        delay_n = np.int(delay[n])
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        if n == np.int(num_conv / row) - 1:
            plt.plot(
                x_values,
                H_true[:, 0, n],
                lw=line_width * 1.2,
                label="$\mathrm{True}$",
                color="k",
            )
        else:
            plt.plot(x_values, H_true[:, 0, n], lw=line_width * 1.2, color="k")
        plt.plot(
            x_values,
            flip[n] * We_init[:, 0, permutations[best_permutation_index][n]],
            lw=line_width * 0.9,
            color="gray",
        )
        if n == np.int(num_conv / row) - 2:
            plt.plot(
                x_values[::k],
                flip[n] * We_init[::k, 0, permutations[best_permutation_index][n]],
                "v",
                markersize=marker_size,
                label="$\mathrm{Initial}$",
                color="gray",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * We_init[::k, 0, permutations[best_permutation_index][n]],
                "v",
                markersize=marker_size,
                color="gray",
            )
        plt.plot(
            x_values[:],
            flip[n] * We_learned[:, 0, permutations[best_permutation_index][n]],
            lw=line_width,
            color="r",
        )
        if n == np.int(num_conv / row) - 3:
            plt.plot(
                x_values[::k],
                flip[n] * We_learned[::k, 0, permutations[best_permutation_index][n]],
                "*",
                markersize=marker_size,
                label="$\mathrm{Learned}$",
                color="r",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * We_learned[::k, 0, permutations[best_permutation_index][n]],
                "*",
                markersize=marker_size,
                color="r",
            )

        if delay_n < 0:
            plt.plot(
                x_values[abs(delay_n) :],
                flip[n]
                * We_learned[:delay_n, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * We_learned[
                        :delay_n:k, 0, permutations[best_permutation_index][n]
                    ],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * We_learned[
                        :delay_n:k, 0, permutations[best_permutation_index][n]
                    ],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        elif delay_n > 0:
            plt.plot(
                x_values[:-delay_n],
                flip[n]
                * We_learned[delay_n:, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * We_learned[
                        delay_n::k, 0, permutations[best_permutation_index][n]
                    ],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * We_learned[
                        delay_n::k, 0, permutations[best_permutation_index][n]
                    ],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        if delay_n == 0:
            plt.plot(
                x_values[:],
                flip[n] * We_learned[:, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * We_learned[::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * We_learned[::k, 0, permutations[best_permutation_index][n]],
                    ".",
                    markersize=marker_size,
                    color="g",
                )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, y_fine)])
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 2:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 3:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 4:
            plt.legend(loc="upper right", ncol=1)

        plt.ylim([y_lim[0], y_lim[1] + 0.05])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{W_{e%i}}$" % (n + 1),
            fontname="Times New Roman",
            fontweight="bold",
        )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/We_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_Wd_sim(
    H_true,
    Wd_init,
    Wd_learned,
    best_permutation_index,
    flip,
    delay,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # normalize all for ploting
    H_true /= np.linalg.norm(H_true, axis=0)
    Wd_init /= np.linalg.norm(Wd_init, axis=0)
    Wd_learned /= np.linalg.norm(Wd_learned, axis=0)

    H_true = np.flip(H_true, axis=0)

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.2 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(Wd_init), np.min(Wd_learned)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(Wd_init), np.max(Wd_learned)]), 2
    )
    if file_number == "2018-08-07-20-05-53":
        y_lim = [-0.5, 0.5]
        y_fine = 0.5

    k = 4
    for n in range(num_conv):
        delay_n = np.int(delay[n])
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        if n == np.int(num_conv / row) - 1:
            plt.plot(
                x_values,
                H_true[:, 0, n],
                lw=line_width * 1.2,
                label="$\mathrm{True}$",
                color="k",
            )
        else:
            plt.plot(x_values, H_true[:, 0, n], lw=line_width * 1.2, color="k")
        plt.plot(
            x_values,
            flip[n] * Wd_init[:, permutations[best_permutation_index][n], 0],
            lw=line_width * 0.9,
            color="gray",
        )
        if n == np.int(num_conv / row) - 2:
            plt.plot(
                x_values[::k],
                flip[n] * Wd_init[::k, permutations[best_permutation_index][n], 0],
                "v",
                markersize=marker_size,
                label="$\mathrm{Initial}$",
                color="gray",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * Wd_init[::k, permutations[best_permutation_index][n], 0],
                "v",
                markersize=marker_size,
                color="gray",
            )
        plt.plot(
            x_values[:],
            flip[n] * Wd_learned[:, permutations[best_permutation_index][n], 0],
            lw=line_width,
            color="r",
        )
        if n == np.int(num_conv / row) - 3:
            plt.plot(
                x_values[::k],
                flip[n] * Wd_learned[::k, permutations[best_permutation_index][n], 0],
                "*",
                markersize=marker_size,
                label="$\mathrm{Learned}$",
                color="r",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * Wd_learned[::k, permutations[best_permutation_index][n], 0],
                "*",
                markersize=marker_size,
                color="r",
            )

        if delay_n < 0:
            plt.plot(
                x_values[abs(delay_n) :],
                flip[n]
                * Wd_learned[:delay_n, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * Wd_learned[
                        :delay_n:k, permutations[best_permutation_index][n], 0
                    ],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * Wd_learned[
                        :delay_n:k, permutations[best_permutation_index][n], 0
                    ],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        elif delay_n > 0:
            plt.plot(
                x_values[:-delay_n],
                flip[n]
                * Wd_learned[delay_n:, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * Wd_learned[
                        delay_n::k, permutations[best_permutation_index][n], 0
                    ],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * Wd_learned[
                        delay_n::k, permutations[best_permutation_index][n], 0
                    ],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        if delay_n == 0:
            plt.plot(
                x_values[:],
                flip[n] * Wd_learned[:, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * Wd_learned[::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * Wd_learned[::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    color="g",
                )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, y_fine)])
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 2:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 3:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 4:
            plt.legend(loc="upper right", ncol=1)

        plt.ylim([y_lim[0], y_lim[1] + 0.05])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{W_{d%i}}$" % (n + 1),
            fontname="Times New Roman",
            fontweight="bold",
        )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/Wd_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_d_sim(
    H_true,
    d_init,
    d_learned,
    best_permutation_index,
    flip,
    delay,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # normalize all for ploting
    H_true /= np.linalg.norm(H_true, axis=0)
    d_init /= np.linalg.norm(d_init, axis=0)
    d_learned /= np.linalg.norm(d_learned, axis=0)

    H_true = np.flip(H_true, axis=0)

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.2 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(d_init), np.min(d_learned)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(d_init), np.max(d_learned)]), 2
    )
    if file_number == "2018-08-07-20-05-53":
        y_lim = [-0.5, 0.5]
        y_fine = 0.5

    k = 4
    for n in range(num_conv):
        delay_n = np.int(delay[n])
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        if n == np.int(num_conv / row) - 1:
            plt.plot(
                x_values,
                H_true[:, 0, n],
                lw=line_width * 1.2,
                label="$\mathrm{True}$",
                color="k",
            )
        else:
            plt.plot(x_values, H_true[:, 0, n], lw=line_width * 1.2, color="k")
        plt.plot(
            x_values,
            flip[n] * d_init[:, permutations[best_permutation_index][n], 0],
            lw=line_width * 0.9,
            color="gray",
        )
        if n == np.int(num_conv / row) - 2:
            plt.plot(
                x_values[::k],
                flip[n] * d_init[::k, permutations[best_permutation_index][n], 0],
                "v",
                markersize=marker_size,
                label="$\mathrm{Initial}$",
                color="gray",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * d_init[::k, permutations[best_permutation_index][n], 0],
                "v",
                markersize=marker_size,
                color="gray",
            )
        plt.plot(
            x_values[:],
            flip[n] * d_learned[:, permutations[best_permutation_index][n], 0],
            lw=line_width,
            color="r",
        )
        if n == np.int(num_conv / row) - 3:
            plt.plot(
                x_values[::k],
                flip[n] * d_learned[::k, permutations[best_permutation_index][n], 0],
                "*",
                markersize=marker_size,
                label="$\mathrm{Learned}$",
                color="r",
            )
        else:
            plt.plot(
                x_values[::k],
                flip[n] * d_learned[::k, permutations[best_permutation_index][n], 0],
                "*",
                markersize=marker_size,
                color="r",
            )

        if delay_n < 0:
            plt.plot(
                x_values[abs(delay_n) :],
                flip[n]
                * d_learned[:delay_n, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * d_learned[:delay_n:k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[abs(delay_n) :: k],
                    flip[n]
                    * d_learned[:delay_n:k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        elif delay_n > 0:
            plt.plot(
                x_values[:-delay_n],
                flip[n]
                * d_learned[delay_n:, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * d_learned[delay_n::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[:-delay_n:k],
                    flip[n]
                    * d_learned[delay_n::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    color="g",
                )
        if delay_n == 0:
            plt.plot(
                x_values[:],
                flip[n] * d_learned[:, permutations[best_permutation_index][n], 0],
                lw=line_width,
                color="g",
            )
            if n == np.int(num_conv / row) - 4:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * d_learned[::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{Learned_{shifted}}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[::k],
                    flip[n]
                    * d_learned[::k, permutations[best_permutation_index][n], 0],
                    ".",
                    markersize=marker_size,
                    color="g",
                )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, y_fine)])
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 2:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 3:
            plt.legend(loc="upper right", ncol=1)
        if n == np.int(num_conv / row) - 4:
            plt.legend(loc="upper right", ncol=1)

        plt.ylim([y_lim[0], y_lim[1] + 0.05])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{d_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/d_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_lambda(
    lambda_init,
    lambda_epochs,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    row=1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = lambda_epochs.shape[0]
    num_conv = lambda_epochs.shape[1]
    # best_epoch = np.argmin(loglambda_loss)

    lambda_epochs = np.concatenate(
        [np.zeros((1, 1)) + lambda_init, lambda_epochs], axis=0
    )

    x_values = np.linspace(0, num_epochs, num_epochs + 1)
    x_lim = [0, num_epochs]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(lambda_epochs))
    y_lim[1] = np.round(1.05 * np.max(lambda_epochs))
    y_lim[0] = 164
    y_lim[1] = 196

    print(np.min(lambda_epochs), lambda_epochs[-1, :])

    # plot
    k = 1
    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        # plt.axhline(y=lambda_init[n], color="k", lw=line_width * 0.5)
        plt.plot(x_values, lambda_epochs[:, n], lw=line_width, color="k")
        plt.plot(x_values[::k], lambda_epochs[::k, n], ".k", lw=line_width)
        # plt.plot(
        #     x_values[best_epoch], lambda_epochs[best_epoch, n], "vb", lw=line_width
        # )

        # for i in range(len(best_val_epoch)):
        #     plt.plot(
        #         x_values[best_val_epoch[i]],
        #         lambda_epochs[best_val_epoch[i], n],
        #         ".k",
        #         lw=line_width,
        #     )

        plt.ylabel("$\lambda$", fontweight="bold")
        plt.xlabel("$\mathrm{Epochs}$", fontweight="bold")
        plt.xticks([i for i in np.arange(x_lim[0], x_lim[1] + 1, 5)])

        if y_lim[1] - y_lim[0] != 0:
            # plt.yticks(
            #     [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 4)]
            # )
            plt.yticks([i for i in np.arange(165, 195 + 1, 10)])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # plt.title(
        #     "$\mathbf{lambda_%i}$" % (n + 1),
        #     fontname="Times New Roman",
        #     fontweight="bold",
        # )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/lambda_{}.pdf".format(PATH, folder_name, file_number)
    )
    plt.savefig(
        "{}/experiments/{}/reports/lambda_{}.eps".format(PATH, folder_name, file_number)
    )


def plot_noiseSTD(
    noiseSTD_epochs,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = noiseSTD_epochs.shape[0]

    # best_epoch = np.argmin(loglambda_loss)
    x_values = np.linspace(1, num_epochs, num_epochs)
    x_lim = [1, num_epochs]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(noiseSTD_epochs))
    y_lim[1] = np.round(1.05 * np.max(noiseSTD_epochs))

    # plot
    k = 1

    ax = fig.add_subplot(111)
    plt.plot(x_values, noiseSTD_epochs, lw=line_width, color="r")
    plt.plot(x_values[::k], noiseSTD_epochs[::k], "vr", lw=line_width)
    plt.plot(x_values[best_epoch], noiseSTD_epochs[best_epoch], "vb", lw=line_width)

    for i in range(len(best_val_epoch)):
        plt.plot(
            x_values[best_val_epoch[i]],
            noiseSTD_epochs[best_val_epoch[i]],
            ".k",
            lw=line_width,
        )

    plt.ylabel("$\mathrm{noiseSTD}$", fontweight="bold")
    plt.xlabel("$\mathrm{Epochs}$", fontweight="bold")
    plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])

    if y_lim[1] - y_lim[0] != 0:
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 4)]
        )
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/noiseSTD_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_H_epochs_real(
    H_init,
    H_learned,
    H_epochs,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_init.shape[2]
    dictionary_dim = H_init.shape[0]
    num_epochs = H_epochs.shape[0]

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.1 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(1.1 * np.min([np.min(H_init), np.min(H_epochs)]), 2)
    y_lim[1] = np.round(1.5 * np.max([np.max(H_init), np.max(H_epochs)]), 2)

    k = 4
    for n in range(num_conv):
        ax = fig.add_subplot(1, num_conv, n + 1)
        for epoch in range(num_epochs):
            plt.plot(x_values, H_epochs[epoch, :, 0, n], lw=line_width, color="b")
            if epoch == num_epochs - 1:
                plt.plot(x_values, H_epochs[epoch, :, 0, n], lw=line_width, color="g")
        plt.plot(x_values, H_init[:, 0, n], lw=line_width, color="gray")
        plt.plot(
            x_values[::k],
            H_init[::k, 0, n],
            "v",
            markersize=marker_size,
            label="$\mathrm{Initial}$",
            color="gray",
        )
        plt.plot(x_values, H_learned[:, 0, n], lw=line_width, color="r")
        plt.plot(
            x_values[::k],
            H_learned[::k, 0, n],
            "*",
            markersize=marker_size,
            label="$\mathrm{Learned}$",
            color="r",
        )
        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.50)])
        plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine)])
        plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
    plt.legend(loc="upper right", ncol=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_epochs_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_H_epochs_sim(
    H_true,
    H_init,
    H_learned,
    H_epochs,
    best_permutation_index,
    flip,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    num_epochs = H_epochs.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.1 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(H_init), np.min(H_epochs)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(H_init), np.max(H_epochs)]), 2
    )

    k = 4
    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        plt.plot(
            x_values,
            H_true[:, 0, n],
            lw=line_width * 1.1,
            label="$\mathrm{True}$",
            color="k",
        )
        plt.plot(
            x_values,
            flip[n] * H_init[:, 0, permutations[best_permutation_index][n]],
            lw=line_width,
            color="gray",
        )
        plt.plot(
            x_values[::k],
            flip[n] * H_init[::k, 0, permutations[best_permutation_index][n]],
            "v",
            markersize=marker_size,
            label="$\mathrm{Initial}$",
            color="gray",
        )
        for epoch in range(num_epochs):
            plt.plot(
                x_values,
                flip[n]
                * H_epochs[epoch, :, 0, permutations[best_permutation_index][n]],
                lw=line_width,
                color="b",
            )
            if epoch == num_epochs - 1:
                plt.plot(
                    x_values,
                    flip[n]
                    * H_epochs[epoch, :, 0, permutations[best_permutation_index][n]],
                    lw=line_width,
                    color="g",
                )
        plt.plot(
            x_values,
            flip[n] * H_learned[:, 0, permutations[best_permutation_index][n]],
            lw=line_width,
            color="r",
        )
        plt.plot(
            x_values[::k],
            flip[n] * H_learned[::k, 0, permutations[best_permutation_index][n]],
            "*",
            markersize=marker_size,
            label="$\mathrm{Learned}$",
            color="r",
        )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine)])
            ax.get_yaxis().set_visible(True)
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.50)])

        plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
    plt.legend(loc="upper right", ncol=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_epochs_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_H_err_epochs_real(
    dist_init_learned_epochs,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    y_fine=0.5,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = dist_init_learned_epochs.shape[1]
    num_conv = dist_init_learned_epochs.shape[0]
    y_values = np.zeros((num_conv, num_epochs))
    y_values = dist_init_learned_epochs

    x_values = np.linspace(0, num_epochs, num_epochs)
    x_lim = [0, (num_epochs)]
    y_lim = [0, 0]
    y_lim[0] = 1.1 * np.min(dist_init_learned_epochs)
    y_lim[1] = 0.9 * np.max(dist_init_learned_epochs)

    # plot
    k = 1
    ax = fig.add_subplot(111)
    c = ["r", "g", "y", "b", "k", "c", "m", "r"]
    marker = ["vr", "*g", ".y", ">b", "xk", "^c", "om", "dr"]
    best_marker = ["vb", "*b", ".b", ">b", "xb", "^b", "ob", "db"]
    best_local_marker = ["vk", "*k", ".k", ">k", "xk", "^k", "ok", "dk"]
    for n in range(num_conv):
        plt.plot(x_values, y_values[n, :], lw=line_width, color=c[n])
        plt.plot(
            x_values[::k],
            y_values[n, ::k],
            marker[n],
            label="$(\mathbf{h}_%i,\hat{\mathbf{h}}_%i)$" % (n + 1, n + 1),
            lw=line_width,
        )

        for i in range(len(best_val_epoch)):
            plt.plot(
                x_values[best_val_epoch[i]],
                y_values[n, best_val_epoch[i]],
                best_local_marker[n],
                lw=line_width,
            )

        plt.plot(
            x_values[best_epoch], y_values[n, best_epoch], best_marker[n], lw=line_width
        )

    plt.ylabel("$\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$")
    plt.xlabel("$\mathrm{Epochs}$")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    if y_lim[1] - y_lim[0] != 0:
        plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine)])
        plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_err_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_H_err_epochs_sim(
    dist_true_learned_epochs,
    dist_true_init,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    y_fine=0.5,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
    output_name="H",
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = dist_true_learned_epochs.shape[1]
    num_conv = dist_true_learned_epochs.shape[0]
    y_values = np.zeros((num_conv, num_epochs + 1))
    y_values[:, 0] = dist_true_init
    y_values[:, 1:] = dist_true_learned_epochs

    x_values = np.linspace(0, num_epochs + 1, num_epochs + 1)
    x_lim = [0, (num_epochs + 1)]
    y_lim = [0, 0]
    y_lim[0] = 1.1 * np.min([np.min(dist_true_learned_epochs), np.min(dist_true_init)])
    y_lim[1] = 0.9 * np.max([np.max(dist_true_learned_epochs), np.max(dist_true_init)])

    # plot
    k = 1
    ax = fig.add_subplot(111)
    c = ["r", "g", "y", "b", "k", "c", "m", "r"]
    marker = ["vr", "*g", ".y", ">b", "xk", "^c", "om", "dr"]
    best_marker = ["vb", "*b", ".b", ">b", "xb", "^b", "ob", "db"]
    best_local_marker = ["vk", "*k", ".k", ">k", "xk", "^k", "ok", "dk"]
    for n in range(num_conv):
        plt.plot(x_values, y_values[n, :], lw=line_width, color=c[n])
        plt.plot(
            x_values[::k],
            y_values[n, ::k],
            marker[n],
            label="$(\mathbf{h}_%i,\hat{\mathbf{h}}_%i)$" % (n + 1, n + 1),
            lw=line_width,
        )

        for i in range(len(best_val_epoch)):
            plt.plot(
                x_values[best_val_epoch[i] + 1],
                y_values[n, best_val_epoch[i] + 1],
                best_local_marker[n],
                lw=line_width,
            )

        plt.plot(
            x_values[best_epoch + 1],
            y_values[n, best_epoch + 1],
            best_marker[n],
            lw=line_width,
        )

        min_err = np.argmin(np.mean(y_values, axis=0))
        plt.plot(x_values[min_err], y_values[n, min_err], ".k", lw=line_width)

    plt.ylabel(
        "$\mathrm{err}(\mathbf{%s}_c,\hat{\mathbf{%s}}_c)$" % (output_name, output_name)
    )
    plt.xlabel("$\mathrm{Epochs}$")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks([i for i in range(0, len(x_values) + 5, 5)])
    if y_lim[1] - y_lim[0] != 0:
        # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine)])
        plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/{}_err_{}.pdf".format(
            PATH, folder_name, output_name, file_number
        )
    )


def plot_H_err_epochs_sim_subplot(
    dist_true_learned_epochs,
    dist_true_init,
    best_epoch,
    best_val_epoch,
    PATH,
    folder_name,
    file_number,
    row=2,
    y_fine=0.5,
    line_width=2.2,
    marker_size=15,
    scale=4,
    scale_height=0.75,
    text_font=45,
    title_font=45,
    axes_font=48,
    legend_font=32,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = dist_true_learned_epochs.shape[1]
    num_conv = dist_true_learned_epochs.shape[0]
    y_values = np.zeros((num_conv, num_epochs + 1))
    y_values[:, 0] = dist_true_init
    y_values[:, 1:] = dist_true_learned_epochs

    x_values = np.linspace(0, num_epochs + 1, num_epochs + 1)
    x_lim = [0, (num_epochs + 1)]
    y_lim = [0, 0]
    # y_lim[0] = 1.1 * np.min([np.min(dist_true_learned_epochs), np.min(dist_true_init)])
    # y_lim[1] = 0.9 * np.max([np.max(dist_true_learned_epochs), np.max(dist_true_init)])
    y_lim[1] = 1
    y_lim[0] = np.min([np.min(dist_true_learned_epochs), np.min(dist_true_init)])

    # plot
    k = 1
    row = 1
    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        plt.plot(x_values, y_values[n, :], lw=line_width, color="k")
        plt.plot(x_values[::k], y_values[n, ::k], ".k", lw=line_width)

        # for i in range(len(best_val_epoch)):
        #     plt.plot(
        #         x_values[best_val_epoch[i] + 1],
        #         y_values[n, best_val_epoch[i] + 1],
        #         best_local_marker[n],
        #         lw=line_width,
        #     )

        plt.plot(
            x_values[best_epoch + 1], y_values[n, best_epoch + 1], "vb", lw=line_width
        )

        if n == (np.int(num_conv / row) - 1):
            plt.plot(
                x_values[best_epoch + 1],
                y_values[n, best_epoch + 1],
                "vb",
                label="Best Val Loss",
                lw=line_width,
            )
            plt.legend(loc="upper right", ncol=1)

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$")
            if y_lim[1] - y_lim[0] != 0:
                plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 2, 6)])

            ax.get_yaxis().set_visible(True)
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
        else:
            plt.xlabel("$\mathrm{Epochs}$")
            plt.xticks([i for i in range(0, len(x_values), 20)])

        plt.title("$\mathbf{h}_%i$" % (n + 1))

        if y_lim[1] - y_lim[0] != 0:
            plt.ylim(y_lim[0], y_lim[1])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_err_subplot_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_denoise_real(
    i,
    y_test,
    y_test_hat,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    a = 10000
    y_test = y_test[:, a : a + 3000]
    y_test_hat = y_test_hat[:, a : a + 3000]

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    input_dim = y_test.shape[1]
    x_values = np.linspace(0, (input_dim * 1000) / sampling_rate, input_dim)
    x_lim = [0, 1.1 * (input_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(y_test[i, :, 0]), np.min(y_test_hat[i, :, 0])])
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(y_test[i, :, 0]), np.max(y_test_hat[i, :, 0])])
    )

    ax = fig.add_subplot(111)

    plt.plot(x_values, y_test[i, :, 0], lw=line_width, label="$y_{noisy}$", color="g")
    plt.plot(x_values, y_test_hat[i, :, 0], lw=line_width, label="$\hat y$", color="r")

    plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
    plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
    plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)])
    plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("$denoising$", fontname="Times New Roman", fontweight="bold")
    plt.legend(loc="upper right", ncol=1)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/denoise_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_denoise_real_2d(
    i,
    y_test,
    y_test_hat,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    ax = fig.add_subplot(121)
    plt.imshow(y_test[i, :, :, 0], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("$input$", fontname="Times New Roman", fontweight="bold")

    ax = fig.add_subplot(122)
    plt.imshow(y_test_hat[i, :, :, 0], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("$output$", fontname="Times New Roman", fontweight="bold")

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/denoise_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_denoise_sim(
    i,
    y_test,
    y_test_noisy,
    y_test_hat,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    input_dim = y_test.shape[1]

    x_values = np.linspace(0, (input_dim * 1000) / sampling_rate, input_dim)
    x_lim = [0, 1.1 * (input_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1
        * np.min(
            [
                np.min(y_test[i, :, 0]),
                np.min(y_test_noisy[i, :, 0]),
                np.min(y_test_hat[i, :, 0]),
            ]
        )
    )
    y_lim[1] = np.round(
        1.5
        * np.max(
            [
                np.max(y_test[i, :, 0]),
                np.max(y_test_noisy[i, :, 0]),
                np.max(y_test_hat[i, :, 0]),
            ]
        )
    )

    ax = fig.add_subplot(111)
    plt.plot(
        x_values, y_test_noisy[i, :, 0], lw=line_width, label="$y_{noisy}$", color="g"
    )
    plt.plot(x_values, y_test[i, :, 0], lw=line_width, label="$y$", color="k")
    plt.plot(x_values, y_test_hat[i, :, 0], lw=line_width, label="$\hat y$", color="r")

    plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
    plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
    plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)])
    # if y_lim[1] - y_lim[0] != 0:
    # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("$denoising$", fontname="Times New Roman", fontweight="bold")
    plt.legend(loc="upper right", ncol=1)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/denoise_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )


def plot_separate_real(
    i,
    spikes,
    y_test_hat_separate,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    y_dim = y_test_hat_separate.shape[1]
    num_conv = z_test_hat.shape[-1]
    x_values = np.linspace(0, (y_dim * 1000) / sampling_rate, y_dim)
    x_lim = [0, 1.1 * (y_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(1.1 * np.min([np.min(y_test_hat_separate[0, :, :])]))
    y_lim[1] = np.round(1.5 * np.max([np.max(y_test_hat_separate[0, :, :])]))

    for n in range(num_conv):
        ax = fig.add_subplot(1, num_conv, n + 1)
        plt.plot(
            x_values,
            y_test_hat_separate[0, (i * dur) : ((i + 1) * dur), n],
            lw=line_width,
            label="$\hat y_{separate}$",
            color="r",
        )

        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.ylim(y_lim)
        plt.xticks(
            [i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
        plt.title(
            "$\mathbf{y_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
        if n == 0:
            plt.legend(loc="upper left", ncol=1)

        plt.legend(loc="upper right", ncol=1)
        fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
        plt.savefig(
            "{}/experiments/{}/reports/separate_{}.pdf".format(
                PATH, folder_name, file_number
            )
        )


def plot_separate_real_series(
    i,
    dur,
    spikes,
    y_series_hat_separate,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = y_series_hat_separate.shape[-1]
    x_values = np.linspace(0, (dur * 1000) / sampling_rate, dur)
    x_lim = [0, 1.1 * (dur * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(y_series_hat_separate[0, (i * dur) : ((i + 1) * dur), :])])
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(y_series_hat_separate[0, (i * dur) : ((i + 1) * dur), :])])
    )

    for n in range(num_conv):
        ax = fig.add_subplot(1, num_conv, n + 1)
        plt.plot(
            x_values,
            y_series_hat_separate[0, (i * dur) : ((i + 1) * dur), n],
            lw=line_width,
            label="$\hat y_{separate}$",
            color="r",
        )

        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.ylim(y_lim)
        plt.xticks(
            [i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )
        plt.yticks(
            [i for i in np.arange(y_lim[0], y_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
        plt.title(
            "$\mathbf{y_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
        if n == 0:
            plt.legend(loc="upper left", ncol=1)

        plt.legend(loc="upper right", ncol=1)
        fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
        plt.savefig(
            "{}/experiments/{}/reports/separate_{}.pdf".format(
                PATH, folder_name, file_number
            )
        )


def plot_code_real(
    i,
    z_test_hat,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    z_dim = z_test_hat.shape[1]
    num_conv = z_test_hat.shape[-1]
    x_values = np.linspace(0, (z_dim * 1000) / sampling_rate, z_dim)
    x_lim = [0, 1.1 * (z_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(1.1 * np.min([np.min(z_test_hat[i, :, :])]))
    y_lim[1] = np.round(1.5 * np.max([np.max(z_test_hat[i, :, :])]))

    for n in range(num_conv):
        ax = fig.add_subplot(1, num_conv, n + 1)
        plt.plot(
            x_values, z_test_hat[i, :, n], lw=line_width, label="$\hat z$", color="r"
        )

        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.xticks(
            [i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)]
        )
        if y_lim[1] - y_lim[0] != 0:
            plt.yticks(
                [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)]
            )
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
        plt.title(
            "$\mathbf{z_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )

        plt.legend(loc="upper right", ncol=1)
        fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
        plt.savefig(
            "{}/experiments/{}/reports/code_{}.pdf".format(
                PATH, folder_name, file_number
            )
        )


def plot_code_real_2d(
    i,
    z_test_hat,
    PATH,
    folder_name,
    file_number,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = z_test_hat.shape[-1]

    k = 4
    for n in range(num_conv):
        if num_conv <= 8:
            ax = fig.add_subplot(1, num_conv, n + 1)
        elif num_conv <= 16:
            ax = fig.add_subplot(2, 8, n + 1)
        elif num_conv <= 32:
            ax = fig.add_subplot(4, 8, n + 1)
        else:
            ax = fig.add_subplot(8, 8, n + 1)
        plt.imshow(z_test_hat[i, :, :, n], cmap="gray")

        plt.xticks([])
        plt.yticks([])
        # plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.title(
            "%i" % (np.sum(np.sum(np.abs(z_test_hat[i, :, :, n])))),
            fontname="Times New Roman",
            fontweight="bold",
        )

        fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
        plt.savefig(
            "{}/experiments/{}/reports/code_{}.pdf".format(
                PATH, folder_name, file_number
            )
        )


def plot_code_sim(
    i,
    z_test,
    z_test_hat,
    best_permutation_index,
    PATH,
    folder_name,
    file_number,
    sampling_rate,
    row=1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    z_dim = z_test.shape[1]
    num_conv = z_test.shape[-1]

    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (z_dim * 1000) / sampling_rate, z_dim)
    x_lim = [0, 1.1 * (z_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(z_test[i, :, :]), np.min(z_test_hat[i, :, :])])
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(z_test[i, :, :]), np.max(z_test_hat[i, :, :])])
    )

    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        plt.plot(x_values, z_test[i, :, n], lw=line_width, label="$z$", color="k")
        plt.plot(
            x_values,
            z_test_hat[i, :, permutations[best_permutation_index][n]],
            lw=line_width,
            label="$\hat z$",
            color="r",
        )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            plt.yticks(
                [i for i in np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 5)]
            )
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)

        if n < np.int(num_conv / row):
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
        else:
            plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
            plt.xticks(
                [i for i in np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 5)]
            )
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1)

        plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{z_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )

    plt.legend(loc="upper right", ncol=1)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/code_{}.pdf".format(PATH, folder_name, file_number)
    )


def plot_alpha_tune_real(
    alpha_list,
    RMSE,
    PATH,
    folder_name,
    file_number,
    y_fine1=0.5,
    y_fine2=0.5,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    x_lim = [np.min(alpha_list), np.max(alpha_list)]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(RMSE))
    y_lim[1] = np.round(1.05 * np.max(RMSE))

    best_index_from_rmse = np.argmin(RMSE)
    second_best_index_from_rmse = RMSE_noisy.index(sorted(RMSE)[1])

    print(
        "best alpha from rmse:",
        alpha_list[np.argmin(RMSE)],
        alpha_list[RMSE_noisy.index(sorted(RMSE)[1])],
    )

    # plot
    k = 1
    ax = fig.add_subplot(111)
    plt.plot(alpha_list, RMSE, lw=line_width, color="k")
    plt.plot(alpha_list[::k], RMSE[::k], "vk", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_rmse],
        RMSE[best_index_from_rmse],
        ".b",
        label="Two Best RMSEs",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_rmse],
        RMSE[second_best_index_from_rmse],
        ".b",
        lw=line_width,
    )
    plt.ylabel("$\mathrm{rmse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\mathrm{alpha}$")

    plt.legend(loc="upper left", ncol=1)
    plt.xticks([i for i in np.arange(np.min(alpha_list), np.max(alpha_list), 1)])
    # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine1)])
    # plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/alpha_tune_real.pdf".format(PATH))


def plot_fwd_alpha_vs_rmse(
    alpha_list,
    RMSE,
    RMSE_noisy,
    noiseSTD,
    PATH,
    folder_name,
    file_number,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=10,
    number_font=15,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    x_lim = [np.min(alpha_list), np.max(alpha_list)]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(RMSE))
    y_lim[1] = np.round(1.05 * np.max(RMSE))

    best_index_from_rmse_true = np.argmin(RMSE)

    print("best alpha from rmse:", alpha_list[best_index_from_rmse_true])

    # plot
    k = 1
    ax = fig.add_subplot(121)
    plt.plot(alpha_list, RMSE, lw=line_width, color="k")
    plt.plot(
        alpha_list[best_index_from_rmse_true],
        RMSE[best_index_from_rmse_true],
        "*b",
        label="Best",
        lw=line_width,
    )
    plt.ylabel("$\mathrm{rmse}(\mathbf{Hx},\hat{\mathbf{y}})$")
    plt.xlabel("$\\alpha$")
    plt.title("$\mathrm{a}$")

    plt.legend(loc="upper left", ncol=1, handletextpad=0.1)
    plt.xticks([i for i in np.round(np.arange(0, 3, 1), 1)])
    plt.yticks([i for i in np.arange(5, 20, 5)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(RMSE_noisy))
    y_lim[1] = np.round(1.05 * np.max(RMSE_noisy))

    ax = fig.add_subplot(122)
    plt.plot(alpha_list, RMSE_noisy, lw=line_width, color="k")
    plt.plot(
        alpha_list[best_index_from_rmse_true],
        RMSE_noisy[best_index_from_rmse_true],
        "*b",
        lw=line_width,
    )
    print(RMSE_noisy[best_index_from_rmse_true])
    # plt.axhline(y=noiseSTD, color='r', lw=line_width*0.2)
    plt.ylabel("$\mathrm{rmse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\\alpha$")
    plt.title("$\mathrm{b}$")

    # plt.legend(loc="upper left", ncol=1, )
    plt.xticks([i for i in np.round(np.arange(0, 3, 1), 1)])
    plt.yticks([i for i in np.arange(15, 30, 5)])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/fwd_alpha_vs_rmse_{}.pdf".format(
            PATH, folder_name, file_number
        )
    )
    plt.savefig(
        "{}/experiments/{}/reports/fwd_alpha_vs_rmse_{}.eps".format(
            PATH, folder_name, file_number
        )
    )


def plot_alpha_tune_sim(
    alpha_list,
    RMSE,
    RMSE_noisy,
    dist_err,
    PATH,
    folder_name,
    file_number,
    y_fine1=0.5,
    y_fine2=0.5,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=10,
    number_font=15,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    x_lim = [np.min(alpha_list), np.max(alpha_list)]
    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(RMSE_noisy))
    y_lim[1] = np.round(1.05 * np.max(RMSE_noisy))
    y_lim = [23, 26]

    best_index_from_rmse_true = np.argmin(RMSE)
    second_best_index_from_rmse_true = RMSE.index(sorted(RMSE)[1])
    best_index_from_rmse = np.argmin(RMSE_noisy)
    second_best_index_from_rmse = RMSE_noisy.index(sorted(RMSE_noisy)[1])
    best_index_from_dict = np.argmin(dist_err)
    second_best_index_from_dict = dist_err.index(sorted(dist_err)[1])

    print(
        "best alpha from rmse noisy:",
        alpha_list[np.argmin(RMSE_noisy)],
        alpha_list[RMSE_noisy.index(sorted(RMSE_noisy)[1])],
    )
    print(
        "best alpha from rmse :",
        alpha_list[np.argmin(RMSE)],
        alpha_list[RMSE.index(sorted(RMSE)[1])],
    )
    print(
        "best alpha from dist err:",
        alpha_list[np.argmin(dist_err)],
        alpha_list[dist_err.index(sorted(dist_err)[1])],
    )

    # plot
    k = 1
    ax = fig.add_subplot(121)
    plt.plot(alpha_list, RMSE_noisy, lw=line_width, color="k")
    plt.plot(alpha_list[::k], RMSE_noisy[::k], "vk", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_rmse],
        RMSE_noisy[best_index_from_rmse],
        "vb",
        label="Best RMSEs",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_rmse],
        RMSE_noisy[second_best_index_from_rmse],
        "vb",
        lw=line_width,
    )
    plt.ylabel("$\mathrm{rmse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\\alpha$")
    plt.title("$\mathrm{a}$")

    plt.legend(loc="upper left", ncol=1)
    plt.xticks(
        [i for i in np.round(np.arange(np.min(alpha_list), np.max(alpha_list), 1), 1)]
    )
    plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 1, 1)])
    plt.ylim(y_lim[0] - 0.5, y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_lim = [0, 0]
    y_lim[0] = np.round(0.95 * np.min(dist_err))
    y_lim[1] = np.round(1.05 * np.max(dist_err))
    y_lim = [0.1, 0.5]

    ax = fig.add_subplot(122)
    plt.plot(alpha_list, dist_err, lw=line_width, color="k")
    plt.plot(alpha_list[::k], dist_err[::k], ".k", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_dict],
        dist_err[best_index_from_dict],
        ".r",
        label="Best Learned Dictionaries",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_dict],
        dist_err[second_best_index_from_dict],
        ".r",
        lw=line_width,
    )
    plt.ylabel(
        "$\\frac{1}{C} \sum_{c=1}^C\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$"
    )
    plt.xlabel("$\\alpha$")
    plt.title("$\mathrm{b}$")

    plt.legend(loc="upper left", ncol=1)
    # plt.xticks(alpha_list)
    plt.xticks(
        [i for i in np.round(np.arange(np.min(alpha_list), np.max(alpha_list), 1), 1)]
    )
    plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.1, 0.1)])
    plt.ylim(y_lim[0] * 0.7, y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/alpha_tune_0_{}.pdf".format(PATH, time))
    plt.savefig("{}/experiments/alpha_tune_0_{}.eps".format(PATH, time))

    fig = newfig(scale=scale, scale_height=scale_height)

    ax = fig.add_subplot(131)
    plt.plot(alpha_list, RMSE_noisy, lw=line_width, color="k")
    plt.plot(alpha_list[::k], RMSE_noisy[::k], "vk", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_rmse],
        RMSE_noisy[best_index_from_rmse],
        ".b",
        label="Two Best RMSEs",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_rmse],
        RMSE_noisy[second_best_index_from_rmse],
        ".b",
        lw=line_width,
    )
    plt.ylabel("$\mathrm{rmse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\mathrm{alpha}$")

    plt.legend(loc="upper left", ncol=1)
    plt.xticks(
        [i for i in np.round(np.arange(np.min(alpha_list), np.max(alpha_list), 1), 1)]
    )
    # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine1)])
    # plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(132)
    plt.plot(alpha_list, RMSE, lw=line_width, color="k")
    plt.plot(alpha_list[::k], RMSE[::k], "vk", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_rmse_true],
        RMSE[best_index_from_rmse_true],
        ".b",
        label="Two Best RMSEs",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_rmse_true],
        RMSE[second_best_index_from_rmse_true],
        ".b",
        lw=line_width,
    )
    plt.ylabel("$\mathrm{rmse}(\mathbf{y},\hat{\mathbf{y}})$")
    plt.xlabel("$\mathrm{alpha}$")

    plt.legend(loc="upper left", ncol=1)
    plt.xticks(
        [i for i in np.round(np.arange(np.min(alpha_list), np.max(alpha_list), 1), 1)]
    )
    # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine1)])
    # plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # y_lim = [0, 0]
    # y_lim[0] = np.round(0.95 * np.min(dist_err))
    # y_lim[1] = np.round(1.05 * np.max(dist_err))

    ax = fig.add_subplot(133)
    plt.plot(alpha_list, dist_err, lw=line_width, color="k")
    plt.plot(alpha_list[::k], dist_err[::k], ".k", lw=line_width)
    plt.plot(
        alpha_list[best_index_from_dict],
        dist_err[best_index_from_dict],
        "vr",
        label="Two Best Dict.",
        lw=line_width,
    )
    plt.plot(
        alpha_list[second_best_index_from_dict],
        dist_err[second_best_index_from_dict],
        "vr",
        lw=line_width,
    )
    plt.ylabel(
        "$\\frac{1}{C} \sum_{c=1}^C\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$"
    )
    plt.xlabel("$\mathrm{alpha}$")

    plt.legend(loc="upper right", ncol=1)
    # plt.xticks(alpha_list)
    plt.xticks(
        [i for i in np.round(np.arange(np.min(alpha_list), np.max(alpha_list), 1), 1)]
    )
    # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1], y_fine2)])
    # plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/alpha_tune_1_{}.pdf".format(PATH, time))


def plot_H_crsae_vs_lcsc(
    H_true,
    H_init,
    H_learned,
    d_learned,
    best_permutation_index_H,
    best_permutation_index_d,
    flip_H,
    flip_d,
    PATH,
    sampling_rate,
    row=1,
    y_fine=0.1,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_true.shape[2]
    dictionary_dim = H_true.shape[0]
    permutations = list(itertools.permutations(np.arange(0, num_conv, 1)))

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.2 * (dictionary_dim * 1000) / sampling_rate]

    y_lim = [0, 0]
    y_lim[0] = np.round(
        1.1 * np.min([np.min(H_true), np.min(H_init), np.min(H_learned)]), 2
    )
    y_lim[1] = np.round(
        1.5 * np.max([np.max(H_true), np.max(H_init), np.max(H_learned)]), 2
    )
    # if file_number == "2018-08-07-20-05-53":
    # y_lim = [-0.5, 0.5]
    # y_fine = 0.5

    k = 4
    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        if n == np.int(num_conv / row) - 1:
            plt.plot(
                x_values,
                H_true[:, 0, n],
                lw=line_width * 4,
                label="$\mathrm{True}$",
                color="k",
            )
        else:
            plt.plot(x_values, H_true[:, 0, n], lw=line_width * 4, color="k")

        plt.plot(
            x_values,
            flip_H[n] * H_init[:, 0, permutations[best_permutation_index_H][n]],
            lw=line_width * 0.9,
            color="gray",
        )

        if n == np.int(num_conv / row) - 2:
            plt.plot(
                x_values[::k],
                flip_H[n] * H_init[::k, 0, permutations[best_permutation_index_H][n]],
                "v",
                markersize=marker_size,
                label="$\mathrm{Init}$",
                color="gray",
            )
        else:
            plt.plot(
                x_values[::k],
                flip_H[n] * H_init[::k, 0, permutations[best_permutation_index_H][n]],
                "v",
                markersize=marker_size,
                color="gray",
            )

        plt.plot(
            x_values[:],
            flip_H[n] * H_learned[:, 0, permutations[best_permutation_index_H][n]],
            lw=line_width,
            color="r",
        )
        plt.plot(
            x_values[:],
            flip_d[n] * d_learned[:, 0, permutations[best_permutation_index_d][n]],
            lw=line_width,
            color="g",
        )
        if n == np.int(num_conv / row) - 3:
            plt.plot(
                x_values[::k],
                flip_H[n]
                * H_learned[::k, 0, permutations[best_permutation_index_H][n]],
                "*",
                markersize=marker_size,
                label="$\mathrm{CRsAE}$",
                color="r",
            )
            plt.plot(
                x_values[::k],
                flip_d[n]
                * d_learned[::k, 0, permutations[best_permutation_index_d][n]],
                ".",
                markersize=marker_size,
                color="g",
            )

        else:
            plt.plot(
                x_values[::k],
                flip_H[n]
                * H_learned[::k, 0, permutations[best_permutation_index_H][n]],
                "*",
                markersize=marker_size,
                color="r",
            )
            if n == 0:
                plt.plot(
                    x_values[::k],
                    flip_d[n]
                    * d_learned[::k, 0, permutations[best_permutation_index_d][n]],
                    ".",
                    markersize=marker_size,
                    label="$\mathrm{LCSC}$",
                    color="g",
                )
            else:
                plt.plot(
                    x_values[::k],
                    flip_d[n]
                    * d_learned[::k, 0, permutations[best_permutation_index_d][n]],
                    ".",
                    markersize=marker_size,
                    color="g",
                )

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
            # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, y_fine)])
            plt.yticks([i for i in np.arange(-0.6, 0.6 + 0.1, 0.6)])
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        # if n < np.int(num_conv / row):
        #     ax.spines["bottom"].set_visible(False)
        #     ax.get_xaxis().set_visible(False)
        #     ax.set_xticklabels([])
        # else:
        #     plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        #     plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.6)])

        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        # plt.xlabel("$\mathrm{Samples\;[n]}$", fontweight="bold")
        plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 0.9)])
        if n == np.int(num_conv / row) - 1:
            plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.3)
        if n == np.int(num_conv / row) - 2:
            plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.05)
        if n == np.int(num_conv / row) - 3:
            plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.05)
        if n == np.int(num_conv / row) - 4:
            plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.05)

        plt.ylim([y_lim[0], y_lim[1] + 0.05])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig("{}/experiments/H.pdf".format(PATH))


def plot_H_err_crsae_vs_lcsc(
    dist_true_learned_epochs,
    dist_true_init,
    dist_true_learned_epochs_2,
    PATH,
    row=2,
    y_fine=0.5,
    line_width=2.5,
    marker_size=20,
    scale=4,
    scale_height=0.75,
    text_font=45,
    title_font=45,
    axes_font=48,
    legend_font=32,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_epochs = dist_true_learned_epochs.shape[1]
    num_conv = dist_true_learned_epochs.shape[0]
    y_values = np.zeros((num_conv, num_epochs + 1))
    y_values[:, 0] = dist_true_init
    y_values[:, 1:] = dist_true_learned_epochs

    y_values_2 = np.zeros((num_conv, num_epochs + 1))
    y_values_2[:, 0] = dist_true_init
    y_values_2[:, 1:] = dist_true_learned_epochs_2

    x_values = np.linspace(0, num_epochs + 1, num_epochs + 1)
    x_lim = [0, (num_epochs + 1)]
    y_lim = [0, 0]
    y_lim[1] = 1
    y_lim[0] = -18

    # plot
    k = 1
    row = 1
    for n in range(num_conv):
        ax = fig.add_subplot(row, np.int(num_conv / row), n + 1)
        plt.plot(x_values, y_values[n, :], lw=line_width, color="k")
        # plt.plot(x_values, y_values_2[n, :], lw=line_width, color="b")
        plt.plot(x_values[::k], y_values[n, ::k], ".k", lw=line_width)

        # if (n == num_conv - 1):
        #     plt.plot(x_values[::k], y_values[n, ::k], ".k", label="CRsAE", lw=line_width)
        #     plt.plot(x_values[::k], y_values_2[n, ::k], "*b", label="LCSC", lw=line_width)
        # else:
        #     plt.plot(x_values[::k], y_values[n, ::k], ".k", lw=line_width)
        #     plt.plot(x_values[::k], y_values_2[n, ::k], "*b", lw=line_width)

        # for i in range(len(best_val_epoch)):
        #     plt.plot(
        #         x_values[best_val_epoch[i] + 1],
        #         y_values[n, best_val_epoch[i] + 1],
        #         best_local_marker[n],
        #         lw=line_width,
        #     )

        # plt.plot(
        #     x_values[best_epoch + 1], y_values[n, best_epoch + 1], "vb", lw=line_width
        # )

        if n == (np.int(num_conv / row) - 1):
            # plt.plot(
            #     x_values[best_epoch + 1],
            #     y_values[n, best_epoch + 1],
            #     "vb",
            #     label="Best Val Loss",
            #     lw=line_width,
            # )
            plt.legend(loc="upper right", ncol=1)

        if n == 0 or n == np.int(num_conv / row):
            plt.ylabel("$\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$")
            if y_lim[1] - y_lim[0] != 0:
                # plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + 0.5, 0.5)])
                plt.yticks([i for i in np.arange(y_lim[0], 0 + 0.5, 6)])
            ax.get_yaxis().set_visible(True)
        else:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticklabels([])

        plt.xlabel("$\mathrm{Epochs}$")
        plt.xticks([i for i in range(0, len(x_values), 10)])
        plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.05)

        plt.title("$\mathbf{h}_%i$" % (n + 1))

        if y_lim[1] - y_lim[0] != 0:
            plt.ylim(y_lim[0], y_lim[1])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0)
    plt.savefig("{}/experiments/H_err_subplot_paper.pdf".format(PATH))


def plot_snr_results(
    snr_list,
    dist_err,
    dist_init_err,
    dist_err_2,
    PATH,
    folder_name,
    file_number,
    y_fine=0.2,
    line_width=2,
    marker_size=15,
    scale=1.2,
    scale_height=1,
    text_font=20,
    title_font=20,
    axes_font=20,
    legend_font=20,
    number_font=20,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    dist_err = np.array(dist_err).T
    dist_init_err = np.array(dist_init_err).T
    dist_err_2 = np.array(dist_err_2).T

    num_conv = dist_err.shape[0]

    x_values = snr_list
    x_lim = [min(x_values), max(x_values)]
    y_values = dist_err
    y_values_2 = dist_err_2
    y_lim = [0, 0]
    y_lim[1] = 0 + 6
    y_lim[0] = np.min(y_values)
    y_lim[0] = -17

    # plot
    k = 1
    ax = fig.add_subplot(111)
    c = ["r", "g", "y", "k"]
    marker = ["vr", "*g", ".y", "ok"]
    c_2 = ["r", "c", "m", "b"]
    marker_2 = ["dr", "^c", "om", "vb"]

    n = 3
    # for n in range(num_conv):
    plt.plot(x_values, y_values[n, :], lw=line_width, color=c[n])
    plt.plot(
        x_values[::k],
        y_values[n, ::k],
        marker[n],
        # label="$(\mathbf{h}_%i,\hat{\mathbf{h}}_%i)_{CRsAE}$" % (n + 1, n + 1),
        label="$\small{CRsAE}$",
        lw=line_width,
    )

    plt.plot(x_values, y_values_2[n, :], lw=line_width, color=c_2[n])
    plt.plot(
        x_values[::k],
        y_values_2[n, ::k],
        marker_2[n],
        # label="$(\mathbf{h}_%i,\hat{\mathbf{h}}_%i)_{LCSC}$" % (n + 1, n + 1),
        label="$\small{LCSC}$",
        lw=line_width,
    )

    y_init = y_values[n, :] * 0 + np.mean(dist_init_err)
    # plt.plot(x_values, y_init, "k", label="$\small{init}$", lw=line_width * 0.4)
    # plt.plot(x_values, y_init, "k", lw=line_width * 0.4)

    plt.ylabel("$\mathrm{err}(\mathbf{h}_%i,\hat{\mathbf{h}}_%i)$" % (n + 1, n + 1))
    plt.xlabel("$\mathrm{SNR [dB]}$")
    plt.xticks([7, 9, 12, 16])
    plt.legend(loc="upper right", ncol=4, columnspacing=0.1, handletextpad=0.05)
    plt.yticks([i for i in np.arange(y_lim[0] + 1, 0 + 2, 4)])
    plt.ylim(y_lim[0], y_lim[1])
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/a_SNR_{}.pdf".format(PATH, time))

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    x_values = snr_list
    x_lim = [min(x_values), max(x_values)]
    x_lim = [6, 20]
    y_values = np.mean(dist_err, axis=0)
    y_values_2 = np.mean(dist_err_2, axis=0)
    y_lim = [0, 0]
    y_lim[0] = 0
    y_lim[1] = np.min(y_values)

    # plot
    k = 1
    ax = fig.add_subplot(111)
    c = "b"
    marker = ".b"

    plt.plot(x_values, y_values, lw=line_width, color=c)
    plt.plot(x_values[::k], y_values[::k], marker, label="$CRsAE$", lw=line_width)

    c_2 = "g"
    marker_2 = "*g"
    plt.plot(x_values, y_values_2, lw=line_width, color=c_2)
    plt.plot(x_values[::k], y_values_2[::k], marker_2, label="$LCSC$", lw=line_width)

    y_init = y_values * 0 + np.mean(dist_init_err)
    plt.plot(x_values, y_init, "k", label="Initial", lw=line_width * 0.4)

    plt.ylabel(
        "$\\frac{1}{C} \sum_{c=1}^C\mathrm{err}(\mathbf{h}_c,\hat{\mathbf{h}}_c)$"
    )
    plt.xlabel("$\mathrm{SNR}$")
    plt.xticks([1.5, 4, 9, 16])
    plt.legend(loc="upper right", ncol=4, columnspacing=0.1, handletextpad=0.05)
    plt.yticks([i for i in np.arange(y_lim[0], y_lim[1] + y_fine, y_fine)])
    plt.ylim(y_lim[0], y_lim[1])
    plt.legend(loc="upper right", ncol=1)
    ax.grid("Off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("{}/experiments/a_SNR_mean_{}.pdf".format(PATH, time))


def plot_H_and_miss_false(
    H_init,
    H_learned,
    missed_list,
    false_list,
    cbp_missed_list,
    cbp_false_list,
    PATH,
    folder_name,
    file_number,
    spikes_filter,
    ch,
    sampling_rate,
    line_width=2,
    marker_size=30,
    scale=4,
    scale_height=0.5,
    text_font=45,
    title_font=55,
    axes_font=48,
    legend_font=34,
    number_font=40,
):

    # upadte plot parameters
    update_plot_parameters(
        text_font=text_font,
        title_font=title_font,
        axes_font=axes_font,
        legend_font=legend_font,
        number_font=number_font,
    )

    # plot new fig
    fig = newfig(scale=scale, scale_height=scale_height)

    num_conv = H_init.shape[2]
    dictionary_dim = H_init.shape[0]

    x_values = np.linspace(0, (dictionary_dim * 1000) / sampling_rate, dictionary_dim)
    x_lim = [0, 1.1 * (dictionary_dim * 1000) / sampling_rate]
    y_lim = [0, 0]
    y_lim[1] = -np.round(1.01 * np.min([np.min(H_init), np.min(H_learned)]), 2)
    y_lim[0] = -np.round(1.15 * np.max([np.max(H_init), np.max(H_learned)]), 2)
    print(y_lim)

    # if file_number == "2019-01-19-13-22-30":
    #     y_lim = [-0.33, 0.7]

    plt.subplots_adjust(wspace=0, hspace=0)
    # make outer gridspec
    outer = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    # make nested gridspecs
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, num_conv, subplot_spec=outer[0], wspace=0.03
    )
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0.25)

    n = 0
    k = 4
    for cell in gs1:
        ax = fig.add_subplot(cell)
        plt.plot(x_values, -H_init[:, 0, n], lw=line_width, color="gray")
        plt.plot(
            x_values[::k],
            -H_init[::k, 0, n],
            "v",
            markersize=marker_size,
            label="$\mathrm{Initial}$",
            color="gray",
        )
        plt.plot(x_values, -H_learned[:, 0, n], lw=line_width, color="r")
        plt.plot(
            x_values[::k],
            -H_learned[::k, 0, n],
            "*",
            markersize=marker_size,
            label="$\mathrm{Learned}$",
            color="r",
        )
        plt.ylabel("$\mathrm{Voltage\;[mV]}$", fontweight="bold")
        plt.xlabel("$\mathrm{Time\;[ms]}$", fontweight="bold")
        plt.xticks([i for i in np.arange(x_lim[0], x_lim[1], 1.5)])
        if n == 0:
            plt.yticks(np.round([-0.25, 0, 0.25, 0.50, 0.75, 1.00], 2))
        plt.ylim([y_lim[0], y_lim[1]])
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n != 0:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.title(
            "$\mathbf{h_%i}$" % (n + 1), fontname="Times New Roman", fontweight="bold"
        )
        n += 1

    plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.01)

    for cell in gs2:
        x_lim = [0, np.round(np.max(missed_list))]
        y_lim = [0, np.round(np.max(false_list))]

        x_lim[1] = 40
        y_lim[1] = 40

        # ax = fig.add_subplot(1, num_conv+1, num_conv+1)
        k = 10
        ax = fig.add_subplot(cell)
        plt.plot(
            missed_list[k:], false_list[k:], lw=line_width, color="green", label="CRsAE"
        )
        plt.plot(
            cbp_missed_list, cbp_false_list, lw=line_width, color="black", label="CBP"
        )
        plt.ylabel("$\mathrm{Estimated\;Spikes\;False\;[\%]}$", fontweight="bold")
        plt.yticks(
            np.round(
                [
                    i
                    for i in np.arange(
                        np.floor(y_lim[0]), np.ceil(10) + 5, (10 - y_lim[0]) / 1
                    )
                ]
            )
        )
        plt.xlabel("$\mathrm{True\;Spikes\;Missed\;[\%]}$", fontweight="bold")
        plt.xticks(
            np.round(
                [
                    i
                    for i in np.arange(
                        np.floor(x_lim[0]),
                        np.ceil(x_lim[1]) + 5,
                        (x_lim[1] - x_lim[0]) / 4,
                    )
                ]
            )
        )
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.legend(loc="upper right", ncol=1, columnspacing=0.1, handletextpad=0.3)
        ax.grid("Off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig(
        "{}/experiments/{}/reports/H_and_miss_false_{}_{}_{}.pdf".format(
            PATH, folder_name, file_number, spikes_filter, ch
        )
    )
