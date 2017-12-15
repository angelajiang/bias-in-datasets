import cv2
import os
import itertools
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
    index = {}
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if num_hists > 10 and debug:
                break
            if name.endswith(suffix):
                image_file = os.path.join(root, name)
                img = cv2.imread(image_file)

                color = ('b','g','r')
                for i,col in enumerate(color):
                    histr = cv2.calcHist([img],[i], None, [256], [0,256])
                    index[name] = histr
                    if debug:
                      plt.plot(histr, color = col)
                      plt.xlim([0,256])
                      plt.savefig("plots/"+col+"-"+name+".png")
                      plt.clf()

                num_hists += 1 

    distances = []
    combos = itertools.combinations(index.values(), 2)
    for combo in combos:
      d = cv2.compareHist(combo[0], combo[1], method=cv2.HISTCMP_BHATTACHARYYA)
      distances.append(d)

    return np.average(distances), np.var(distances)


if __name__ == "__main__":
    dataset_path = "/datasets/BigLearning/ahjiang/image-data/imagenet/"
    suffix = "JPEG"

    dataset_path = "/home/angela/src/data/image-data/train/train_images_subset/"
    suffix = "jpg"
    avg_distance, var_distance  = get_histograms(dataset_path, suffix, True)
    print avg_distance, var_distance
