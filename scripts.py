import csv 

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")
def plot_exposure_csv(csv_file):
    ys = []
    errs = []
    labels = []
    with open(csv_file, "rb") as f:
        for line in f:
            vals = line.split(",")
            name = vals[0]
            avg = float(vals[1])
            stddev = float(vals[2])
            ys.append(avg)
            errs.append(stddev)
            labels.append(name)
    xs = range(len(ys))
    plt.bar(xs, ys, yerr=errs)
    plt.xticks(xs, labels)
    plt.ylabel("Avg distance between color hists", fontsize=10)
    plt.savefig("plots/exposure.pdf")

if __name__ == "__main__":
    plot_exposure_csv("output/exposure.csv")
