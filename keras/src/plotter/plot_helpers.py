"""
Copyright (c) 2019 CRISP

Plot helpers.

:author: Bahareh Tolooshams
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

PATH = ".."


def newfig(scale, scale_height=1, square=False):
    """
    Use this function to initialize an image to plot on
    """
    plt.clf()
    fig = plt.figure(figsize=figsize(scale, scale_height, square))
    return fig


def savefig(folder_name, filename):
    """
    Use this function to save a formatted pdf of the current plot
    """
    plt.savefig("{}/experiments/{}/reports/{}.pdf".format(PATH, folder_name, filename))


def figsize(scale, scale_height=1, square=False):
    """
    This functions defines a figure size with golden ratio, or a square figure size that fits a letter size paper
    figsize(1) will return a fig_size whose width fits a 8 by 11 letter size paper, and height determined by the golde ratio.
    If for some reason you want to strech your plot vertically, use the scale_height argument to 1.5 for example.
    figsize(1, square=True) returns a square figure size that fits a letter size paper.
    """
    fig_width_pt = 416.83269  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    if square:
        fig_size = [fig_width, fig_width]
    else:
        fig_height = fig_width * golden_mean * scale_height  # height in inches
        fig_size = [fig_width, fig_height]
    return fig_size


def update_plot_parameters(
    text_font=50, title_font=50, axes_font=46, legend_font=39, number_font=40
):
    """
    Helper function update plot paramters
    :param text_font: text_font
    :param title_font: title_font
    :param axes_font: axes_font
    :param legend_font: legend_font
    :param number_font: number_font
    :return: none
    """
    sns.set_style("whitegrid")
    sns.set_context("poster")

    ## These handles changing matplotlib background to unify fonts and fontsizes
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [
            "Times"
        ],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": axes_font,  # LaTeX default is 10pt font.
        "axes.titlesize": title_font,
        "font.size": text_font,
        "legend.fontsize": legend_font,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": number_font,
        "ytick.labelsize": number_font,
        "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage{fontspec}",
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ],
    }

    mpl.rcParams.update(pgf_with_latex)
