import os
import json
import operator
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot(filepath, plotdir):
    data = {}
    acc_tuples = []
    loss_tuples = []
    best_trial = None
    with open(filepath) as f:
        for line in f:
            vals = line.split(",")
            trial = int(vals[1])
            spline_y1 = float(vals[2])
            spline_y2 = float(vals[3])
            spline_y3 = float(vals[4])
            loss = float(vals[5])
            acc = float(vals[6])
            data[trial] = {"ys":[spline_y1, spline_y2, spline_y3]}
            data[trial]["loss"] = loss
            data[trial]["acc"] = acc
            acc_tuples.append((trial, acc))
            loss_tuples.append((trial, loss))

    sorted_trial_acc = sorted(acc_tuples, key=lambda x: x[1], reverse=True)
    sorted_trial_loss = sorted(loss_tuples, key=lambda x: x[1])

    acc_plotdir = "{}/acc".format(plotdir)
    if not os.path.exists(acc_plotdir):
        os.makedirs(acc_plotdir)

    loss_plotdir = "{}/loss".format(plotdir)
    if not os.path.exists(loss_plotdir):
        os.makedirs(loss_plotdir)

    xs = [0, 0.5, 1]

    for i, (trial, acc) in enumerate(sorted_trial_acc):
        f = interp1d(xs, data[trial]["ys"], kind="quadratic")
        plt.plot(xs, f(xs))
        plt.ylim(0, 1.1)
        plt.title(acc)
        plotfile = "{}/acc-{}-trial{}.png".format(acc_plotdir, i, trial)
        plt.savefig(plotfile)
        plt.clf()

    for i, (trial, loss) in enumerate(sorted_trial_loss):
        f = interp1d(xs, data[trial]["ys"], kind="quadratic")
        plt.plot(xs, f(xs))
        plt.ylim(0, 1.1)
        plt.title(loss)
        plotfile = "{}/loss-{}-trial{}.png".format(loss_plotdir, i, trial)
        plt.savefig(plotfile)
        plt.clf()

def plot_old(filepath, plotdir, v1=True):
    data = {}
    acc_tuples = []
    loss_tuples = []
    best_trial = None
    with open(filepath) as f:
        for line in f:
            vals = line.split(" ")
            if "start_trial" in vals:
                if v1:
                    trial = int(vals[2]) - 1
                    spline_y1 = round(float(vals[4][:-1]), 6)
                    spline_y2 = round(float(vals[6][:-1]), 6)
                    spline_y3 = round(float(vals[8][:-1]), 6)
                    print("--------------------")
                    print("Trial: {}, spline_y1: {}, spline_y2: {}, spline_y3: {}".format(trial,
                                                                                          spline_y1,
                                                                                          spline_y2,
                                                                                          spline_y3))
                    data[trial] = {"ys":[spline_y1, spline_y2, spline_y3]}
                else:
                    trial = int(vals[2]) - 1
                    data[trial] = {}
            elif "trial_params" in vals:
                spline_y1 = round(float(vals[2][:-1]), 6)
                spline_y2 = round(float(vals[4][:-1]), 6)
                spline_y3 = round(float(vals[6][:-2]), 6)
                print("--------------------")
                print("Trial: {}, spline_y1: {}, spline_y2: {}, spline_y3: {}".format(trial,
                                                                                      spline_y1,
                                                                                      spline_y2,
                                                                                      spline_y3))
                data[trial] = {"ys":[spline_y1, spline_y2, spline_y3]}
            elif "trial_test_loss" in vals:
                loss = float(vals[1])
                tmp = vals[4]
                acc = float(tmp.rstrip()[:-1])
                print("Loss: {}, Acc: {}".format(loss, acc))
                data[trial]["loss"] = loss
                data[trial]["acc"] = acc
                acc_tuples.append((trial, acc))
                loss_tuples.append((trial, loss))
            elif "best_study_result" in vals:
                best_trial = int(vals[2][:-1])

    print("Best trial:", best_trial)
    sorted_trial_acc = sorted(acc_tuples, key=lambda x: x[1], reverse=True)
    sorted_trial_loss = sorted(loss_tuples, key=lambda x: x[1])

    acc_plotdir = "{}/acc".format(plotdir)
    if not os.path.exists(acc_plotdir):
        os.makedirs(acc_plotdir)

    loss_plotdir = "{}/loss".format(plotdir)
    if not os.path.exists(loss_plotdir):
        os.makedirs(loss_plotdir)

    xs = [0, 0.5, 1]

    for i, (trial, acc) in enumerate(sorted_trial_acc):
        f = interp1d(xs, data[trial]["ys"])
        plt.plot(xs, f(xs))
        plt.ylim(0, 1.1)
        plt.title(acc)
        plotfile = "{}/acc-{}-trial{}.png".format(acc_plotdir, i, trial)
        plt.savefig(plotfile)
        plt.clf()

    for i, (trial, loss) in enumerate(sorted_trial_loss):
        f = interp1d(xs, data[trial]["ys"])
        plt.plot(xs, f(xs))
        plt.ylim(0, 1.1)
        plt.title(loss)
        plotfile = "{}/loss-{}-trial{}.png".format(loss_plotdir, i, trial)
        plt.savefig(plotfile)
        plt.clf()


if __name__ == "__main__":

    expnames = ["200223-acc-random", "200223-acc-gpy"]
    primes = [1]
    betas = [1]
    seeds = [4444]
    for expname in expnames:
        for prime in primes:
            for beta in betas:
                for seed in seeds:
                    print("\n============== {} ===============".format(expname))
                    filename = "data/{}/sb-{}prime-{}beta-{}seed-summary.txt".format(expname, prime, beta, seed)
                    plotdir = "/Users/angela/src/private/bias-in-datasets/active_learning/plots/{}".format(expname)
                    if not os.path.exists(plotdir):
                        os.makedirs(plotdir)
                    plot(filename, plotdir)

