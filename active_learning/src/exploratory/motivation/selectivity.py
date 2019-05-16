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

    leg = plt.legend(loc=0, prop={'size': label_size * .8})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

def selectivity_lines(filenames, labels, epochs, style, plot_dir):
    fraction_same_epoch = []
    cosine_sim_epoch = []
    selectivities = []

    for i, (epoch, filename, label) in enumerate(zip(epochs, filenames, labels)):
        fraction_same_epoch.append([])
        cosine_sim_epoch.append([])
        with open(filename, "rb") as f:
            data = pickle.load(f)
            raw_cosine_sims = data["cos_sims"]
            raw_fraction_same = data["fraction_same"]

            # Aggregate data from batches, into data by selectivity
            fraction_same_selectivity = {}
            for fraction_same_batch in raw_fraction_same:
                for k, d in fraction_same_batch.iteritems():
                    if k not in fraction_same_selectivity.keys():
                        fraction_same_selectivity[k] = []
                    if round(k, 1) not in selectivities:
                        selectivities.append(round(k,1))
                    fraction_same_selectivity[k].append(d)

            # Aggregate data from batches, into data by selectivity
            for k in sorted(fraction_same_selectivity.keys()):
                d = fraction_same_selectivity[k]
                average_fraction_same = np.mean(d)
                fraction_same_epoch[i].append(average_fraction_same)

            # Aggregate data from batches, into data by selectivity
            cosine_sim_selectivity = {}
            for cosine_sim_batch in raw_cosine_sims:
                for k, d in cosine_sim_batch.iteritems():
                    if k not in cosine_sim_selectivity.keys():
                        cosine_sim_selectivity[k] = []
                    cosine_sim_selectivity[k].append(d)

            # Aggregate data from batches, into data by selectivity
            for k in sorted(cosine_sim_selectivity.keys()):
                d = cosine_sim_selectivity[k]
                average_cosine_sim = np.mean(d)
                cosine_sim_epoch[i].append(average_cosine_sim)

    for epoch in range(len(cosine_sim_epoch)):
        xs = []
        ys = []
        for selectivity in range(len(cosine_sim_epoch[epoch])):
            xs.append((selectivity + 1) / 10.)
            ys.append(cosine_sim_epoch[epoch][selectivity])
        if epoch % 2 == 0:
            plt.plot(xs, ys, marker="o", linestyle=style, label="{}-Epoch {}0".format(label, epoch))
    plt.xlabel("Selectivity")
    plt.ylabel("Cosine Similarity")
    plt.legend()

def selectivity_colorfield(filenames, labels, epochs, plot_dir):
    fraction_same_epoch = []
    cosine_sim_epoch = []
    selectivities = []

    for i, (epoch, filename, label) in enumerate(zip(epochs, filenames, labels)):
        fraction_same_epoch.append([])
        cosine_sim_epoch.append([])
        with open(filename, "rb") as f:
            data = pickle.load(f)
            raw_cosine_sims = data["cos_sims"]
            raw_fraction_same = data["fraction_same"]

            # Aggregate data from batches, into data by selectivity
            fraction_same_selectivity = {}
            for fraction_same_batch in raw_fraction_same:
                for k, d in fraction_same_batch.iteritems():
                    if k not in fraction_same_selectivity.keys():
                        fraction_same_selectivity[k] = []
                    if round(k, 1) not in selectivities:
                        selectivities.append(round(k,1))
                    fraction_same_selectivity[k].append(d)

            # Aggregate data from batches, into data by selectivity
            for k in sorted(fraction_same_selectivity.keys()):
                d = fraction_same_selectivity[k]
                average_fraction_same = np.mean(d)
                fraction_same_epoch[i].append(average_fraction_same)

            # Aggregate data from batches, into data by selectivity
            cosine_sim_selectivity = {}
            for cosine_sim_batch in raw_cosine_sims:
                for k, d in cosine_sim_batch.iteritems():
                    if k not in cosine_sim_selectivity.keys():
                        cosine_sim_selectivity[k] = []
                    cosine_sim_selectivity[k].append(d)

            # Aggregate data from batches, into data by selectivity
            for k in sorted(cosine_sim_selectivity.keys()):
                d = cosine_sim_selectivity[k]
                average_cosine_sim = np.mean(d)
                cosine_sim_epoch[i].append(average_cosine_sim)

    fraction_same_matrix = np.asarray(fraction_same_epoch).transpose()
    cosine_sim_matrix = np.asarray(cosine_sim_epoch).transpose()

    pp.pprint(cosine_sim_epoch)
    pp.pprint(fraction_same_matrix)

    im = plt.imshow(fraction_same_matrix, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of Examples Chosen")
    plt.xticks(range(len(epochs)), epochs)
    plt.yticks(range(len(selectivities)), sorted(selectivities))
    plt.title("Fraction of Gradient Weights W/ Same Sign")
    filename = os.path.join(plot_dir, "fracsame_heatmap")
    plt.savefig("{}.png".format(filename))
    plt.savefig("{}.pdf".format(filename))
    plt.clf()


    im = plt.imshow(cosine_sim_matrix, vmin=-1, vmax=1, cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of Examples Chosen")
    plt.xticks(range(len(epochs)), epochs)
    plt.yticks(range(len(selectivities)), sorted(selectivities))
    plt.title("Cosine Similarity Compared to NoFilter Gradient")

    filename = os.path.join(plot_dir, "cossim_heatmap")
    plt.show()
    #plt.savefig("{}.png".format(filename))
    #plt.savefig("{}.pdf".format(filename))
    plt.clf()


if __name__ == "__main__":
    expname1 = "190510_gradlog"
    expname2 = "190510_gradlog_random"
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    subdir1 = "{}/pickles/biases/".format(expname1)
    subdir2 = "{}/pickles/biases/".format(expname2)
    plot_dir = "/Users/angela/src/private/bias-in-datasets/active_learning/plots/{}".format(expname1)
    baseline = "sampling_cifar10_mobilenetv2_1_64_0.0_0.0005_trial1_seed1337_biases"

    prefix1 = os.path.join(home, subdir1, baseline)
    prefix2 = os.path.join(home, subdir2, baseline)

    if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    exps = [
            (prefix1, "TopLoss", "-"),
            (prefix2, "Random", ":"),
            ]

    potential_epochs = range(0, 111, 10)
    potential_epochs = range(0, 91, 10)
    for prefix, label, style in exps:
        filenames = []
        labels = []
        epochs = []
        for epoch in potential_epochs:
            filename = prefix+".epoch_{}.pickle".format(epoch)
            if os.path.isfile(filename):
                filenames.append(filename)
                labels.append(label)
                epochs.append(epoch)
        #selectivity_colorfield(filenames, labels, epochs, plot_dir)
        selectivity_lines(filenames, labels, epochs, style, plot_dir)
    format_plot("Selectivity", "Cosine Similarity")
    filename = os.path.join(plot_dir, "cossim_lines")
    plt.savefig("{}.png".format(filename))
    plt.savefig("{}.pdf".format(filename))
    print(filename)

    colors = [BASELINE_COLOR, SB_COLOR, COLOR1, COLOR2]


