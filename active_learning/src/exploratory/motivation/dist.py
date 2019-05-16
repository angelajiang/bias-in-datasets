import pickle
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

def colorfield(filenames, labels, epoch):
    for filename, label in zip(filenames, labels):
        with open(filename, "rb") as f:
            d = pickle.load(f)
            losses = d["losses"] # 391, xs
            baseline_norms = d["baseline_norms"] # 391 x 173
            dists = d["dists"] # 391 x 173
            ys = range(len(d["dists"][0]))
            matrix = []
            for i, (bnorms, ds) in enumerate(zip(d["baseline_norms"], d["dists"])):
                matrix.append([])
                for jvar, (bnorm, d) in enumerate(zip(bnorms, ds)):
                    relative_dist = d / float(bnorm)
                    matrix[i].append(relative_dist)
            plt.imshow(matrix, cmap='viridis')
            plt.show()

def selectivity_v_dist(filenames, labels, epoch, plot_dir):
    print("Plotting selectivity_v_dist for epoch {}".format(epoch))
    d_fraction_same = {}
    d_selectivities = {}

    for filename, label in zip(filenames, labels):
        with open(filename, "rb") as f:
            d = pickle.load(f)
            baseline_norms = d["baseline_norms"] # 391 x 173
            dists = d["cos_sims"] # 391 x 173
            selectivities = d["selectivities"] # 391
            cosine_sims = d["cos_sims"]
            fraction_same = d["fraction_same"]
            average_dists = [np.mean(cs) for cs in cosine_sims]
            average_fraction_same = [np.mean(fs) for fs in fraction_same]
            d_fraction_same[label] = average_fraction_same
            d_selectivities[label] = selectivities
            plt.scatter(selectivities, average_dists, label=label)
    plt.xlabel("Fraction Selected for Training")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.legend()
    plotfile = os.path.join(plot_dir, "selectivity_v_dist_epoch{}.png".format(epoch))
    plt.savefig(plotfile)
    plt.clf()

    for label, fraction_same in d_fraction_same.iteritems(): 
        selectivities = d_selectivities[label]
        plt.scatter(selectivities, fraction_same, label=label)
    plt.xlabel("Fraction Selected for Training")
    plt.ylabel("Fraction of Gradient Weights w/ Same Sign")
    plt.ylim(0, 1)
    plt.legend()
    plotfile = os.path.join(plot_dir, "selectivity_v_fraction_same_epoch{}.png".format(epoch))
    plt.savefig(plotfile)
    plt.clf()

if __name__ == "__main__":
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    subdir = "{}/pickles/biases/".format("190508_gradlog")
    plot_dir = "/Users/angela/src/private/bias-in-datasets/active_learning/plots/{}".format("190508_gradlog")
    sb = "sampling_cifar10_mobilenetv2_0.1_32_0.0_0.0005_trial1_seed1337_biases"
    topk6 = "topk_cifar10_mobilenetv2_6_32_0.0_0.0005_trial1_seed1337_biases"
    topk8 = "topk_cifar10_mobilenetv2_8_32_0.0_0.0005_trial1_seed1337_biases"
    topk16 = "topk_cifar10_mobilenetv2_16_32_0.0_0.0005_trial1_seed1337_biases"
    topk24= "topk_cifar10_mobilenetv2_24_32_0.0_0.0005_trial1_seed1337_biases"


    exps = [
            (topk6, "3/16"),
            (topk8, "1/4"),
            (topk16, "1/2"),
            (topk24, "3/4"),
            (sb, "SB"),
            ]

    #epochs = [1, 2, 3, 4, 5, 6]
    epochs = range(30, 101, 5)
    for epoch in epochs:
        filenames = []
        labels = []
        for f, label in exps:
            filename = os.path.join(home, subdir, f+".epoch_{}.pickle".format(epoch))
            if os.path.isfile(filename):
                filenames.append(filename)
                labels.append(label)
        #colorfield(filenames, labels, epoch)
        selectivity_v_dist(filenames, labels, epoch, plot_dir)

    colors = [BASELINE_COLOR, SB_COLOR, COLOR1, COLOR2]


