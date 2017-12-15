import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import sys
import util

def split_data_by_channel(data):
    assert len(data.shape) == 4 # b, h, w, c
    return data[:, :, :, 0], data[:, :, :, 1], data[:, :, :, 2]

def generate_all_histograms(data):
    hists = []
    for i, img in enumerate(data):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = (hist / np.sum(hist)).flatten()
        hists.append(hist)
    return hists

def generate_pairwise_histogram_distances(hists, num_samples=1000000):
    def combos(j):
        idx = np.random.choice(np.arange(len(hists)), (j, 2))
        for row in idx:
            yield row
    distances = []
    viz_step = int(num_samples / 20)
    for i, combo in enumerate(combos(num_samples)):
        if i % viz_step == 0: print i
        hist1, hist2 = hists[combo[0]], hists[combo[1]]
        d = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_BHATTACHARYYA)
        distances.append(d)
    return distances

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    data = util.load_cifar10(dataset_path)
    hists = generate_all_histograms(data)
    distances = generate_pairwise_histogram_distances(hists)

    data_r, data_g, data_b = split_data_by_channel(data)
    print("RGB Average: [%f, %f, %f]" % \
          (np.mean(data_r), np.mean(data_g), np.mean(data_b)))
    print("RGB Stddev: [%f, %f, %f]" % \
          (np.std(data_r), np.std(data_g), np.std(data_b)))
    print("Average Histogram distance: %f" % np.mean(distances))
    print("Histogram distance stddev: %f" % np.std(distances))
