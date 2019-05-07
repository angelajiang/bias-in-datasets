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

def momentum_v_backprops(filenames, labels):
    for filename, label in zip(filenames, labels):
        backprops = []
        momentums = []
        with open(filename) as f:
            for line in f:
                if "average_momentum" in line:
                    vals = line.strip().split(" ")
                    momentum = float(vals[1])
                    momentums.append(momentum)
                if "train_debug" in line:
                    vals = line.strip().split(",")
                    bp = int(vals[2])
                    backprops.append(bp)
        plt.plot(backprops, momentums, label=label)
    plt.xlabel("Images Backpropped")
    plt.ylabel("Momentum")
    plt.yscale("log")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    baseline = "sampling_cifar10_mobilenetv2_1_32_0.0_0.0005_trial1_seed1337_v2"
    sb = "sampling_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    dynbatch = "dynbatch_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_v2"
    sblr64 = "sblr_cifar10_mobilenetv2_0.1_64_0.0_0.0005_trial1_seed1337_v2"
    exps = [
            ("190425_sblr", baseline, "Baseline"),
            ("190425_sblr", sb, "SB"),
            #("190425_sblr", dynbatch, "Dynamic Batching + SBLR"),
            #("190425_sblr", sblr64, "Batchsize 64 + SBLR"),
            ]

    filenames = []
    labels = []
    for exp, f, label in exps:
        filename = os.path.join(home, exp, f)
        filenames.append(filename)
        labels.append(label)

    momentum_v_backprops(filenames, labels)

