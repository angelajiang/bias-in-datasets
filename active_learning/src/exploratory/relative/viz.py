import os
import numpy as np
import math
import matplotlib.pyplot as plt
import ast

import math
import ast
import ast
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_palette("husl")

BASELINE_COLOR ="#009e73"
SB_COLOR ="#cc79a7"
COLOR1 ="#0072b2"
COLOR2 ="#d55e00"
COLOR3 ="#0072b2"

def format_plot(xlabel, ylabel, label_size=11, grid=False):
    plt.tick_params(axis='y', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='y', which='minor', labelsize=label_size * 1.2)
    plt.tick_params(axis='x', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='x', which='minor', labelsize=label_size * 1.2)

    plt.xlabel(xlabel, fontsize=label_size * 1.6)
    plt.ylabel(ylabel, fontsize=label_size * 1.6)
    plt.tight_layout()
    plt.gca().xaxis.grid(grid)
    plt.gca().yaxis.grid(grid)

    leg = plt.legend(loc=0, prop={'size': label_size * 1.4})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

def plot_batchsize(filename, label):
    with open(filename) as f:
        last_num_backprops = 0
        batch_sizes = []
        for line in f:
            if "train_debug" in line:
                vals = line.split(",")
                backprops = int(vals[2])
                batch_size = backprops - last_num_backprops
                last_num_backprops = backprops
                batch_sizes.append(batch_size)
    batches = range(len(batch_sizes))
    plt.scatter(batches, batch_sizes, label=label)
    plt.ylim(0, max(batch_sizes) + 16)
    plt.xlabel("Number of batches")
    plt.ylabel("Batch size")
    plt.legend()
    plt.show()

def accuracy_v_sb_backprops(filenames, labels, colors):
    fig = plt.figure()

    for filename, label, color in zip(filenames, labels, colors):
        if "SBLR" in label:
            bp_idx = 7
        else:
            bp_idx = 2
        backprops = []
        accuracies = []
        with open(filename) as f:
            last_num_backprops = 0
            batch_sizes = []
            for line in f:
                if "test_debug" in line:
                    vals = line.split(",")
                    bp = int(vals[bp_idx]) / 1000000.
                    acc = float(vals[5])
                    backprops.append(bp)
                    accuracies.append(100 - acc)
        plt.plot(backprops, accuracies, label=label, c=color)
    format_plot("Images SB would have bp (millions)", "Test Error Percent ")
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.NullFormatter())
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.legend()
    plt.show()

def accuracy_v_backprops(filenames, labels):
    for filename, label in zip(filenames, labels):
        backprops = []
        accuracies = []
        with open(filename) as f:
            last_num_backprops = 0
            batch_sizes = []
            for line in f:
                if "test_debug" in line:
                    vals = line.split(",")
                    bp = int(vals[2]) / 1000000.
                    acc = float(vals[5])
                    backprops.append(bp)
                    accuracies.append(100 - acc)
        plt.plot(backprops, accuracies, label=label)
    format_plot("Images Backpropped (millions)", "Error")
    plt.yscale("log")
    plt.legend()
    plt.show()

def make_instantaneous(l):
    lcopy = l[:]
    lcopy.insert(0, 0)        
    pairs = zip(lcopy[::1], lcopy[1::1])
    ilist = [j - i for i, j in pairs]
    return ilist

def selectivity_v_backprops(filenames, labels):
    for filename, label in zip(filenames, labels):
        backprops = []
        skips = []
        with open(filename) as f:
            last_num_backprops = 0
            batch_sizes = []
            for line in f:
                if "test_debug" in line:
                    vals = line.split(",")
                    bp = int(vals[2]) / 1000000.
                    s = int(vals[3]) / 1000000.
                    backprops.append(bp)
                    skips.append(s)
        ibps = make_instantaneous(backprops)
        iss = make_instantaneous(skips)
        ratios = [ibp / float(s + ibp) for ibp, s  in zip(ibps[1:], iss[1:])]
        plt.plot(backprops[1:], ratios, label=label)
    format_plot("Images Backpropped (millions)", "Ratio Backpropped")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"

    baseline = "sampling_cifar10_mobilenetv2_1_128_0.0_0.0005_trial1_seed1337_v2"
    sb = "sampling_cifar10_mobilenetv2_0.1_128_0.0_0.0005_trial1_seed1337_v2"
    relative1024 = "sampling_cifar10_mobilenetv2_0_128_1024_0.0005_trial1_seed1337_v2"
    relative50000 = "sampling_cifar10_mobilenetv2_0_128_50000_0.0005_trial1_seed1337_v2"
    kath = "kath_cifar10_mobilenetv2_1024_128_0.0_0.0005_trial1_seed1337_v2"

    exps = [
            ("150403_baseline", baseline, "No Filter"),
            ("150403_baseline", sb, "Selective-Backprop"),
            ("190503_relative_cross", relative1024, "Relative-SB-1024"),
            ("190507_relative-squared_cross", relative1024, "Relative-Squared-SB-1024"),
            ("190507_relative-cubed_cross", relative1024, "Relative-Cubed-SB-1024"),
            #("190504_kath", kath, "Katharopoulos18 Sampling"),
            ]

    exps = [
            ("190509_icml", baseline, "No Filter"),
            ("190509_icml", sb, "Selective-Backprop"),
            ("190509_relative-cubed_cross", relative1024, "Relative-Cubed-SB-1024"),
            #("190504_kath", kath, "Katharopoulos18 Sampling"),
            ]

    filenames = []
    labels = []
    for exp, f, label in exps:
        filename = os.path.join(home, exp, f)
        filenames.append(filename)
        labels.append(label)
    colors = [BASELINE_COLOR, SB_COLOR, COLOR1, COLOR2]

    accuracy_v_backprops(filenames, labels)
    selectivity_v_backprops(filenames, labels)

