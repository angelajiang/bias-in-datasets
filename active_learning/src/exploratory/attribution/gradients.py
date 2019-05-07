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


def gradient_v_backprops(inputs, labels):
    new_losses = []
    empiricals = []
    for input_file, label in zip(inputs, labels):
        total_norm = 0
        xs = []
        ys = []
        with open(input_file) as f:
            for line in f:
                if "total_norm" in line:
                    vals = line.split(" ")
                    norm = float(vals[1])
                    total_norm += norm
                if "train_debug" in line:
                    vals = line.split(",")
                    backprops = int(vals[2]) / 1000000.
                    xs.append(backprops)
                    ys.append(norm)
        plt.plot(xs, ys, label=label, alpha=0.6)
    plt.xlabel("Num backpropped (millions)")
    plt.ylabel("Gradients of batch")
    plt.ylim(0, 2.5)

    plt.legend()
    plt.show()

def error_v_backprops(inputs, labels):
    new_losses = []
    empiricals = []
    for input_file, label in zip(inputs, labels):
        total_norm = 0
        xs = []
        ys = []
        with open(input_file) as f:
            for line in f:
                if "total_norm" in line:
                    vals = line.split(" ")
                    norm = float(vals[1])
                    total_norm += norm
                if "test_debug" in line:
                    vals = line.split(",")
                    acc = float(vals[5])
                    backprops = int(vals[2]) / 1000000.
                    xs.append(backprops)
                    ys.append(100 - acc)
        plt.plot(xs, ys, label=label)
    plt.xlabel("Num backpropped (millions)")
    plt.ylabel("Error")
    plt.yscale("log")

    plt.legend()
    plt.show()

def error_v_gradients(inputs, labels):
    new_losses = []
    empiricals = []
    for input_file, label in zip(inputs, labels):
        with open(input_file) as f:
            total_norm = 0
            xs = []
            ys = []
            for line in f:
                if "total_norm" in line:
                    vals = line.split(" ")
                    norm = float(vals[1])
                    total_norm += norm
                if "test_debug" in line:
                    vals = line.split(",")
                    acc = float(vals[5])
                    xs.append(total_norm)
                    ys.append(100 - acc)
        plt.plot(xs, ys, label=label)
    #plt.xlim(0, 50000)
    plt.xlabel("Total gradients applied")
    plt.ylabel("Error")
    plt.yscale("log")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    inputs = []
    labels = []
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    nofilter = "sampling_cifar10_mobilenetv2_1_128_0.0_0.0005_trial1_seed1337_v2"
    nofilter196 = "sampling_cifar10_mobilenetv2_1_196_0.0_0.0005_trial1_seed1337_v2"
    nofilter224 = "sampling_cifar10_mobilenetv2_1_224_0.0_0.0005_trial1_seed1337_v2"
    nofilter256 = "sampling_cifar10_mobilenetv2_1_256_0.0_0.0005_trial1_seed1337_v2"
    nofilter512 = "sampling_cifar10_mobilenetv2_1_512_0.0_0.0005_trial1_seed1337_v2"
    sb = "sampling_cifar10_mobilenetv2_0.1_128_0.0_0.0005_trial1_seed1337_v2"
    sb512 = "sampling_cifar10_mobilenetv2_0.1_512_0.0_0.0005_trial1_seed1337_v2"

    exps = [
    ("190417_norm", sb, "SB"),
    ("190417_norm", nofilter, "NoFilter"),
    #("190417_cross_reweighted", nofilter, "Reweighted"),
    #("190417_batch196_cross_reweighted", nofilter196, "Reweighted-196"),
    #("190417_batch196_3x_cross_reweighted", nofilter196, "Reweighted-196-3x"),
    #("190417_batch224_3x_cross_reweighted", nofilter224, "Reweighted-224-3x"),
    #("190417_batch240_cross_reweighted", nofilter224, "Reweighted-224"),
    #("190417_batch240_cross", nofilter224, "NoFilter-224"),
    #("190417_batch256_3x_cross_reweighted", nofilter256, "Reweighted-256-3x"),
    ("190417_batch512_cross_reweighted", nofilter512, "Reweighted-512"),
    ("190417_batch512_cross", nofilter512, "NoFilter-512"),
    ("190417_batch512_cross", sb512, "SB-512"),
    ]

    for exp, f, label in exps:
        a = os.path.join(home, exp, f)
        inputs.append(a)
        labels.append(label)

    error_v_gradients(inputs, labels)
    #gradient_v_backprops(inputs, labels)
    error_v_backprops(inputs, labels)
