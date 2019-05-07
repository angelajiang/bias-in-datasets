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
                    bp = int(vals[2])
                    acc = float(vals[5])
                    backprops.append(bp)
                    accuracies.append(acc)
        plt.plot(backprops, accuracies, label=label)
    plt.xlabel("Images Backpropped")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"

    baseline = "sampling_cifar10_mobilenetv2_1_32_0.0_0.0005_trial1_seed1337_v2"
    sb = "sampling_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    dynbatch = "dynbatch_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    sblr64 = "sblr_cifar10_mobilenetv2_0.1_64_0.0_0.0005_trial1_seed1337_v2"
    reweight_dynbatch = "reweight-dynbatch_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    reweight32 = "reweight_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    upweight = "upweight_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    exps = [
            ("190423_fast", baseline, "No Filter"),
            ("190423_fast", sb, "Selective-Backprop (Ours)"),
            #("190425_sblr", upweight, "SB + upweight"),
            #("190425_sblr", dynbatch, "Dynamic Batching + SBLR"),
            ("190423_fast", reweight_dynbatch, "No Filter + Reweight + Dynamic Batching"),
            #("190424_sblr", reweight_dynbatch, "Reweight +  Dynamic Batching + SBLR"),
            #("190425_sblr", sblr64, "Batchsize 64 + SBLR"),
            ]

    filenames = []
    labels = []
    for exp, f, label in exps:
        filename = os.path.join(home, exp, f)
        #plot_batchsize(filename, label)
        filenames.append(filename)
        labels.append(label)
    colors = [BASELINE_COLOR, SB_COLOR, COLOR1]

    accuracy_v_backprops(filenames, labels)
    accuracy_v_sb_backprops(filenames, labels, colors)

