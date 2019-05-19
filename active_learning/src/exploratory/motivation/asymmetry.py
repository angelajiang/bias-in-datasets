
import pickle
import pprint as pp
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

    leg = plt.legend(loc=0, prop={'size': label_size * 1.2})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)


def plot(filenames, labels, net, gpu, plot_dir):
    avg_forwards = []
    avg_calcs = []
    avg_updates = []
    std_forwards = []
    std_calcs = []
    std_updates = []
    for filename in filenames:
        forwards = []
        calcs = []
        updates = []
        first_backwards = True
        first_update = True
        with open(filename) as f:
            for line in f:
                if "forwards" in line:
                    vals = line.rstrip().split(" ")
                    forward = float(vals[1])
                    forwards.append(forward)
                elif "backwards" in line:
                    if not first_backwards:
                        vals = line.rstrip().split(" ")
                        calcs.append(float(vals[1]))
                    else:
                        first_backwards = False
                elif "update" in line:
                    if not first_update:
                        vals = line.rstrip().split(" ")
                        if vals[0] == "update":
                            updates.append(float(vals[1]))
                    else:
                        first_update = False
        avg_forwards.append(np.average(forwards))
        avg_calcs.append(np.average(calcs))
        avg_updates.append(np.average(updates))
        std_forwards.append(np.std(forwards))
        std_calcs.append(np.std(calcs))
        std_updates.append(np.std(updates))
    width = 0.35
    ind = np.arange(len(avg_forwards))

    bottom_sum = [a + b for a, b in zip(avg_forwards, avg_calcs)]
    bp_sum = [a + b for a, b in zip(avg_updates, avg_calcs)]

    ratios = [float(bp) / f for f, bp in zip(avg_forwards, bp_sum)]

    p1 = plt.bar(ind, avg_forwards, width, yerr=std_forwards)
    p2 = plt.bar(ind, avg_calcs, bottom=avg_forwards, width=width, yerr=std_calcs)
    p3 = plt.bar(ind, avg_updates, bottom=bottom_sum, width=width, yerr=std_updates)

    plt.xticks(ind, labels)

    plt.legend((p1[0], p2[0], p3[0]), ("Forwards", "Backwards-Calculate", "Backwards-Update"))
    format_plot("Batch size", "Seconds")
    plot_file = os.path.join(plot_dir, "{}_{}".format(gpu, net))
    plt.ylim(0, .4)
    plt.tight_layout()
    plt.savefig(plot_file + ".png", bbox_inches = "tight")
    plt.savefig(plot_file + ".pdf", bbox_inches = "tight")
    plt.clf()
    return ratios

if __name__ == "__main__":
    homedir = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/microbenchmarks"
    plot_homedir = "/Users/angela/src/private/bias-in-datasets/active_learning/plots"
    #nets = ["mobilenetv2"]
    nets = ["resnet", "mobilenetv2"]
    nets = ["mobilenetv2"]
    batch_sizes = [1, 32, 64, 128]
    tuples = [
            ("190517_asymmetry_titanv", "TitanV"),
            ("190517_asymmetry_GTX1070", "GTX-1070"),
            ("190517_asymmetry_k20", "K20"),
            ]

    all_ratios = []
    for expname, gpu in tuples:
        filenames = []
        labels = []
        plot_dir = os.path.join(plot_homedir, "asymmetry", expname)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for net in nets:
            for batch_size in batch_sizes:
                fname = "asymmetry_{}_{}".format(net, batch_size)
                fpath = os.path.join(homedir, expname, fname)
                if os.path.exists(fpath):
                    label = "{}".format(batch_size)
                    filenames.append(fpath)
                    labels.append(label)
            ratios = plot(filenames, labels, net, gpu, plot_dir)
            print(ratios)
            all_ratios.append((gpu, net, ratios))

    plot_dir = os.path.join(plot_homedir, "asymmetry", "190517_asymmetry")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for gpu, net, ratios in all_ratios:
        print ratios
        plt.plot(ratios, marker="o", label="{} on {}".format(net, gpu))
    plt.ylim(0, 3)
    plt.xticks(range(len(ratios)), batch_sizes)
    format_plot("Batch size", "Backwards / forwards latency")
    plot_file = os.path.join(plot_dir, "asymmetry")
    plt.tight_layout()
    plt.savefig(plot_file + ".png")
    plt.savefig(plot_file + ".pdf")
    plt.clf()

