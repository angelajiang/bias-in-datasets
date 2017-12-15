import cv2
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def get_histograms(dataset_path, suffix, num_channels=3, debug=False):
    num_hists = 0
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
                    if debug:
                      plt.plot(histr, color = col)
                      plt.xlim([0,256])
                      plt.savefig("plots/"+col+"-"+name+".png")
                      plt.clf()

                num_hists += 1 
    return 0,0


if __name__ == "__main__":
    dataset_path = "/datasets/BigLearning/ahjiang/image-data/imagenet/"
    suffix = "JPEG"

    dataset_path = "/home/angela/src/data/image-data/train/train_images_subset/"
    suffix = "jpg"
    avg, var = get_histograms(dataset_path, suffix, True)
