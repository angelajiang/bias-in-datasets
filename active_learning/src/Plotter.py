import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import metrics


def write_file(plot_file_prefix, show=False):

    plot_file = "{}.pdf".format(plot_file_prefix)
    plt.savefig(plot_file)
    print(plot_file)

    plot_file = "{}.png".format(plot_file_prefix)
    plt.savefig(plot_file, format="png", dpi=1000)

    if show:
        plt.show()
    plt.clf()

def format_plot(xlabel, ylabel, label_size=10, legend_scale = 0.9, grid=False):
    plt.tick_params(axis='y', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='y', which='minor', labelsize=label_size * 1.2)
    plt.tick_params(axis='x', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='x', which='minor', labelsize=label_size * 1.2)

    plt.xlabel(xlabel, fontsize=label_size * 1.6)
    plt.ylabel(ylabel, fontsize=label_size * 1.6)
    plt.tight_layout()
    plt.gca().xaxis.grid(grid)
    plt.gca().yaxis.grid(grid)

    leg = plt.legend(loc=0, prop={'size': label_size * legend_scale})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

def format_plot_2ys(ax1, ax2, xlabel, ylabel1, ylabel2, label_size=10, legend_scale = 0.9, grid=False):
    ax1.tick_params(axis='y', which='major', labelsize=label_size * 1.4)
    ax1.tick_params(axis='y', which='minor', labelsize=label_size * 1.2)
    ax1.tick_params(axis='x', which='major', labelsize=label_size * 1.4)
    ax1.tick_params(axis='x', which='minor', labelsize=label_size * 1.2)

    ax2.tick_params(axis='y', which='major', labelsize=label_size * 1.4)
    ax2.tick_params(axis='y', which='minor', labelsize=label_size * 1.2)
    ax2.tick_params(axis='x', which='major', labelsize=label_size * 1.4)
    ax2.tick_params(axis='x', which='minor', labelsize=label_size * 1.2)

    ax1.set_xlabel(xlabel, fontsize=label_size * 1.6)
    ax1.set_ylabel(ylabel1, fontsize=label_size * 1.6)
    ax2.set_ylabel(ylabel2, fontsize=label_size * 1.6)

    plt.tight_layout()
    plt.gca().xaxis.grid(grid)
    plt.gca().yaxis.grid(grid)

    leg = ax1.legend(loc=2, prop={'size': label_size * legend_scale})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    leg = ax2.legend(loc=4, prop={'size': label_size * legend_scale})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)


def find_first_x_at_y(xs, ys, ymarker):
    if ymarker is None:
        return None
    for x, y in zip(xs, ys):
        if y >= ymarker:
            return x
    return None
