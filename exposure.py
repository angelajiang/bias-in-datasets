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

def get_histograms(dataset_path, suffix, num_channels=3, debug=False):
    num_hists = 0
    debug_num_hists = 10
    plot_dir = "plots"
    index = {}

    for root, dirs, files in os.walk(dataset_path):
        if num_hists > debug_num_hists and debug:
            break

        for name in files:
            if num_hists > debug_num_hists and debug:
                break

            if name.endswith(suffix):
                image_file = os.path.join(root, name)
                img = cv2.imread(image_file)

                color = ('b','g','r')
                for i,col in enumerate(color):
                    histr = cv2.calcHist([img],[i], None, [256], [0,256])
                    index[name] = histr
                    if debug:
                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)
                        plt.plot(histr, color = col)
                        plt.xlim([0,256])
                        plt.savefig(os.path.join(plot_dir, col+"-"+name))
                        plt.clf()

                num_hists += 1 

    combos = itertools.combinations(index.values(), 2)

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

    avg_distance, var_distance  = get_histograms(dataset_path, suffix, debug=True)
    print dataset_path, avg_distance, var_distance
