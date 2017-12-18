import cv2
import os
import itertools
import math
import numpy as np
import util
import random

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")


#cv2.HISTCMP_CORREL
#cv2.HISTCMP_CHISQR
#cv2.HISTCMP_INTERSECT 
#cv2.HISTCMP_BHATTACHARYYA

def plot_histogram(histr, plot_dir, name):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.plot(histr)
    #plt.xlim([0,256])
    plt.savefig(os.path.join(plot_dir, name + ".jpg"))
    plt.clf()

def plot_color_histograms(hists, plot_dir, name):
    colors = ["r", "g", "b"]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for i, c in enumerate(colors):
        plt.plot(hists[i], color=c)
    plt.savefig(os.path.join(plot_dir, name + ".jpg"))
    plt.clf()

def get_image_histograms(dataset_path, suffix, num_channels, total_num_hists):

    num_hists = 0

    for root, dirs, files in os.walk(dataset_path):
        if num_hists > total_num_hists:
            break

        for name in files:
            if num_hists > total_num_hists:
                break

            if name.endswith(suffix):
                image_file = os.path.join(root, name)
                img = cv2.imread(image_file)
                colors = ["b", "g", "r"]
                hists = []
                for i, color in enumerate(colors):
                    histr = cv2.calcHist([img],
                                         [i],
                                         None,
                                         [256],
                                         [0,256])
                    histr = (histr / np.sum(histr)).flatten()
                    hists.append(histr)

                yield tuple(hists)

                if num_hists % 100  == 0:
                    print "Created {} histograms".format(num_hists)

                num_hists += 1 

def get_class_histograms(dataset_path, suffix, num_channels, num_classes, num_hists_per_class):
    all_class_directories = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]
    random.shuffle(all_class_directories)
    class_directories = all_class_directories[:num_classes]
    class_histograms = []

    for d in class_directories:
        hist_gen = get_image_histograms(d, suffix, num_channels, num_hists_per_class)
        class_histograms.append(hist_gen)

    return class_histograms
    

def plot_dataset_histogram(dataset_path, suffix, text_name, num_channels, num_classes, num_hists_per_class):

    all_histograms = get_class_histograms(dataset_path, suffix, num_channels, num_classes, num_hists_per_class)

    dataset_hist_r = None
    dataset_hist_g = None
    dataset_hist_b = None
    for class_hists_gen in all_histograms:
        for hists in class_hists_gen:
            if dataset_hist_r is None:
                dataset_hist_r, dataset_hist_g, dataset_hist_b = hists
                plot_histogram(dataset_hist_r, "plots", "hr")
                plot_histogram(dataset_hist_g, "plots", "hg")
                plot_histogram(dataset_hist_b, "plots", "hb")
            else:
                dataset_hist_r += hists[0]
                dataset_hist_g += hists[1]
                dataset_hist_b += hists[2]

    plot_histogram(dataset_hist_r, "plots", "class-hist-r")
    plot_histogram(dataset_hist_b, "plots", "class-hist-b")
    plot_histogram(dataset_hist_g, "plots", "class-hist-g")
    plot_color_histograms([dataset_hist_r, dataset_hist_g, dataset_hist_b], "plots", text_name + "-hist")

def get_histogram_distances(dataset_path, suffix, num_channels, num_classes, num_hists_per_class):

    class_histograms = get_class_histograms(dataset_path, suffix, num_channels, num_classes, num_hists_per_class)
    all_histograms = itertools.chain.from_iterable(class_histograms)
    combos = itertools.combinations(all_histograms, 2)

    n = 0.
    sum_x = 0.
    sum_xx = 0.

    for combo in combos:
        h1 = combo[0][0] + combo[0][1] + combo[0][2]
        h2 = combo[1][0] + combo[1][1] + combo[1][2]
        d = cv2.compareHist(h1, h2, method=cv2.HISTCMP_BHATTACHARYYA)
        n += 1
        sum_x += d
        sum_xx += d * d

    avg = sum_x / n
    std_dev = math.sqrt((n * sum_xx - sum_x * sum_x)/(n * (n - 1)))

    return avg, std_dev

if __name__ == "__main__":
    dataset_path = "/home/angela/src/data/image-data/train/train_images_subset/"
    suffix = "jpg"

    dataset_path = "/datasets/BigLearning/ahjiang/image-data/training/oxford/"
    suffix = "jpg"

    dataset_path = "/datasets/BigLearning/ahjiang/image-data/imagenet/"
    suffix = "JPEG"
    name = "imagenet"
    plot_dataset_histogram(dataset_path, suffix, name, 3, 10, 10)
    avg, std_dev = get_histogram_distances(dataset_path, suffix, 3, 10, 10)
    print dataset_path, avg, std_dev

    dataset_path = "/datasets/BigLearning/ahjiang/image-data/cifar/cifar-10-batches-py" 
    data, labels = util.load_cifar10(dataset_path)
