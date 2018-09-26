#!/usr/bin/env python
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import os

sns.set_style("white")


class Config:
    MATCHER = ("^(.*)_(.*)_(\d+)_(\d+)_(\d*\.?\d*)_(\d*\.?\d*)(_trial\d+)?")

    def __init__(self, filename):
        groups = self.matches(filename)
        self.dataset = groups[0]
        self.network = groups[1]
        self.top_k = int(groups[2])
        self.pool_size = int(groups[3])
        self.lr = float(groups[4])
        self.decay = float(groups[5])
        self.trial = None
        if groups[6]:
            unparsed_trial = groups[6]
            self.trial = unparsed_trial.strip("_trial")

    def matches(self, filename):
        import re
        return (re.match(Config.MATCHER, filename)).groups()


class LineResult:
    def __init__(self, line_type, epoch, num_backprop, loss, time, acc):
        self.line_type = line_type
        self.epoch = epoch
        self.num_backprop = num_backprop
        self.loss = loss
        self.time = time
        self.acc = acc

    @property
    def is_train(self):
        return self.line_type == "train_debug"

    @property
    def is_test(self):
        return self.line_type == "test_debug"


def parse_line_v1(line):
    vals = line.split(',')
    if "train_debug" in line:
        epoch = int(vals[1])
        num_backprop = int(vals[2])
        loss = float(vals[4])
        time = float(vals[5])
        acc = float(vals[6])
        line_type = "train_debug"
    elif "test_debug" in line:
        epoch = int(vals[1])
        num_backprop = int(vals[2])
        loss = float(vals[3])
        acc = float(vals[4])
        time = float(vals[5])
        line_type = "test_debug"
    else:
        return None
    return LineResult(line_type, epoch, num_backprop, loss, time, acc)


def parser_for(filename):
    version = filename.split('_')[-1]
    if version == "v1":
        return parse_line_v1
    else:
        Exception("Version cannot be {}".format(version))


def parse_file(filename):
    parser = parser_for(filename)
    with open(filename) as f:
        parsed = [parser(line)
                  for line in f]
    return ([d for d in parsed if d and d.is_train],
            [d for d in parsed if d and d.is_test])


def plot(xs_by_config, ys_by_config, xlabel, ylabel, plot_dir, smoothing=0):
    for config, ys in sorted(ys_by_config.iteritems(), key=lambda x: x[0].top_k):

        xs = xs_by_config[config]
        if len(xs) == 0:
            continue

        for i in range(smoothing):
            xs = [(l+r) / 2. for l, r in zip(xs[:-1], xs[1:])]
            ys = [(l+r) / 2. for l, r in zip(ys[:-1], ys[1:])]

        if config.trial:
            label = "top_{}/{}, lr={}, decay={} trial-{}".format(config.top_k,
                                                                 config.pool_size,
                                                                 config.lr,
                                                                 config.decay,
                                                                 config.trial)
        else:
            label = "top_{}/{}, lr={}, decay={}".format(config.top_k,
                                                        config.pool_size,
                                                        config.lr,
                                                        config.decay)
        print xlabel, ",", ylabel, len(xs), len(ys)
        if config.top_k == config.pool_size:
            plt.plot(xs, ys, label=label, linestyle="--", linewidth=1)
        else:
            plt.plot(xs, ys, label=label, linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plot_file = "{}_{}_{}_top{}_ps{}_lr{}.pdf".format(config.dataset,
                                                      ylabel,
                                                      config.network,
                                                      config.top_k,
                                                      config.pool_size,
                                                      config.lr)
    plot_file = "{}_{}_{}_top{}_ps{}_lr{}.png".format(config.dataset,
                                                      ylabel,
                                                      config.network,
                                                      config.top_k,
                                                      config.pool_size,
                                                      config.lr)

    plt.legend()
    plt.savefig(os.path.join(plot_dir, plot_file))
    plt.clf()


def plot_distributions(pickles_dir, plot_dir):
    for filename in os.listdir(pickles_dir):
        config = Config(filename)
        pickles_file = os.path.join(pickles_dir, filename)
        with open(pickles_file, 'rb') as handle:
            d = pickle.load(handle)
            n, bins, patches = plt.hist(d.values(),
                                        50,
                                        normed=0,
                                        facecolor='green',
                                        alpha=0.75)

            plt.xlim(0, max(d.values()) + 1)
            plt.xlabel("Num backprops")
            plt.ylabel("Num images")

            subplot_dir = os.path.join(plot_dir, "images_dist")
            if not os.path.isdir(subplot_dir):
                os.mkdir(subplot_dir)
            plot_file = os.path.join(subplot_dir, filename + ".pdf")
            plt.savefig(plot_file)
            plt.clf()


def plot_experiment(experiment_dir, plot_dir, smoothing=0):
    train_num_backprops_by_config = {}
    test_num_backprops_by_config = {}
    train_losses_by_config = {}
    test_losses_by_config = {}
    train_raw_times_by_config = {}
    test_raw_times_by_config = {}
    train_accuracies_by_config = {}
    test_accuracies_by_config = {}

    pickles_dir = os.path.join(experiment_dir, "pickles")
    if os.path.isdir(pickles_dir):
        plot_distributions(pickles_dir, plot_dir)

    for filename in os.listdir(experiment_dir):
        if filename == ".DS_Store":
            continue
        elif filename == "pickles":
            continue

        filepath = os.path.join(experiment_dir, filename)
        config = Config(filename)
        train_lines, test_lines = parse_file(filepath)

        train_num_backprops = [l.num_backprop for l in train_lines]
        test_num_backprops = [l.num_backprop for l in test_lines]
        train_raw_times = [l.time for l in train_lines]
        test_raw_times = [l.time for l in test_lines]
        train_losses = [l.loss for l in train_lines]
        test_losses = [l.loss for l in test_lines]
        train_accuracies = [l.acc for l in train_lines]
        test_accuracies = [l.acc for l in test_lines]

        train_num_backprops_by_config[config] = train_num_backprops
        test_num_backprops_by_config[config] = test_num_backprops
        train_raw_times_by_config[config] = train_raw_times
        test_raw_times_by_config[config] = test_raw_times
        train_losses_by_config[config] = train_losses
        test_losses_by_config[config] = test_losses
        train_accuracies_by_config[config] = train_accuracies
        test_accuracies_by_config[config] = test_accuracies


    plot(train_num_backprops_by_config, train_losses_by_config, "Num Images Trained", "Training Loss", plot_dir, smoothing)
    plot(test_num_backprops_by_config, test_accuracies_by_config, "Num Images Trained", "Test Accuracy", plot_dir, smoothing)
    plot(test_num_backprops_by_config, test_losses_by_config, "Num Images Trained", "Test Loss", plot_dir, smoothing)
    #plot(train_raw_times_by_config, train_losses_by_config, "Wall Clock Time", "Training Loss By Time", plot_dir)
    #plot(accuracies_by_config, "Wall Clock Time", "Test Accuracy By Time", plot_dir, xs_by_config=test_times_by_config)
    #plot(test_losses_by_config, "Wall Clock Time", "Test Loss By Time", plot_dir, xs_by_config=test_times_by_config)


def main():
    plot_home_dir = "plots"

    # Resnet basic with image histograms (no 0 bar)
    experiment_name = "180915c_cifar10_resnet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir)

    # Resnet with no duplicates
    experiment_name = "180916b_cifar10_resnet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir, 3)

    # MobileNet with no duplicates
    experiment_name = "180916b_cifar10_mobilenet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir, 3)

    # MobileNet with 250000 backprops
    experiment_name = "180916c_cifar10_mobilenet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir, 250)

    # ResNet with 250000 backprops
    experiment_name = "180916c_cifar10_resnet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir, 250)

    # MNIST basic
    experiment_name = "180916_mnist"
    experiment_dir = "data/output/mnist/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_experiment(experiment_dir, plot_dir)

    # MobileNet with 250000 backprops
    experiment_name = "180917_cifar10"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_experiment(experiment_dir, plot_dir, 250)

if __name__ == "__main__":
    main()


