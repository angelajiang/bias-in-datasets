import cv2
import os
import itertools
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


#cv2.HISTCMP_CORREL
#cv2.HISTCMP_CHISQR
#cv2.HISTCMP_INTERSECT 
#cv2.HISTCMP_BHATTACHARYYA

def plot_histogram(histr, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.plot(histr, color = col)
    plt.xlim([0,256])
    plt.savefig(os.path.join(plot_dir, col+"-"+name))
    plt.clf()

def get_image_histograms(dataset_path, suffix, num_channels, debug):
    num_hists = 0
    debug_num_hists = 100

    for root, dirs, files in os.walk(dataset_path):
        if num_hists > debug_num_hists and debug:
            break

        for name in files:
            if num_hists > debug_num_hists and debug:
                break

            if name.endswith(suffix):
                image_file = os.path.join(root, name)
                img = cv2.imread(image_file)
                histr = cv2.calcHist([img],
                                     range(num_channels),
                                     None,
                                     [8,8,8],
                                     [0,256,0,256,0,256])
                yield histr

                if num_hists % 1000 == 0:
                    print "Created {} histograms".format(num_hists)

                num_hists += 1 


def get_histogram_distances(dataset_path, suffix, num_channels=3, debug=False):

    histograms = get_image_histograms(dataset_path, suffix, num_channels, debug)
    combos = itertools.combinations(histograms, 2)

    n = 0.
    sum_x = 0.
    sum_xx = 0.

    for combo in combos:
        d = cv2.compareHist(combo[0], combo[1], method=cv2.HISTCMP_BHATTACHARYYA)
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

    avg_distance, var_distance  = get_histogram_distances(dataset_path, suffix, debug=True)
    print dataset_path, avg_distance, var_distance
