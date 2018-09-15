#!/usr/bin/env python
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import os

sns.set_style("white")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_pickle_file(title, pickle_dir):
    pickle_file = os.path.join(pickle_dir, "{}.pickle".format(title))
    return pickle_file

def pickle_scalar(events_file, scalar, title, pickle_dir):
    vals = []
    for e in tf.train.summary_iterator(events_file):
        for v in e.summary.value:
            if v.tag == scalar:
                vals.append(v.simple_value)
                if len(vals) % 1000 == 0:
                    pickle_file = get_pickle_file(title, pickle_dir)
                    print "Pickling {} with {} vals...".format(pickle_file, len(vals))
                    with open(pickle_file, 'wb') as handle:
                        pickle.dump(vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print "Done pickling."

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def plot_scalar(events_file, scalar, title, plot_dir, pickle_dir, force_pickle=False, num_batches=5):
    pickle_file = get_pickle_file(title, pickle_dir)
    if not os.path.exists(pickle_file) or force_pickle:
        #pickle_scalar(events_file, scalar, title, pickle_dir)
        try:
            pickle_scalar(events_file, scalar, title, pickle_dir)
        except:
            pass

    with open(pickle_file, 'rb') as handle:
        vals = pickle.load(handle)

    batches = []
    batch_size = len(vals) / num_batches
    min_bin = float("inf")
    max_bin = 0
    for x in get_batch(vals, batch_size):
        batches.append(x)
        if max(x) > max_bin:
            max_bin = int(max(x))
        if min(x) < min_bin:
            min_bin = int(min(x))

    #fig = plt.figure(figsize=(40, 16))
    rows = 1
    cols = len(batches)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), sharex=True, sharey=True)

    num_bins = 500

    for i in range(len(batches)):

        ax = axes[i]
        batch = batches[i]

        print "{}: {} vals, {} bins".format(title, len(batch), num_bins)
        n, bins, patches = ax.hist(batch, num_bins, normed=0, facecolor='green', alpha=0.75, range=(min_bin, max_bin))
        #plt.title(title, fontsize=15)
        #plt.xlabel(scalar, fontsize=15)
        #plt.ylabel("Frequency", fontsize=15)

    plt.xscale("log")
    plt.tight_layout()
    plot_file = os.path.join(plot_dir, "{}-split{}.png".format(title, num_batches))
    plt.savefig(plot_file)
    plt.clf()

class Config:
    def __init__(self, filename):
        vals = filename.split("_")
        self.dataset = vals[0]
        self.network = vals[1]
        self.top_k = int(vals[2])
        self.pool_size = int(vals[3])
        self.lr = float(vals[4][:-3])

class LineResult:
    def __init__(self, line_type, epoch, num_backprop, loss, time, acc):
        self.line_type = line_type
        self.epoch = epoch
        self.num_backprop = num_backprop
        self.loss = loss
        self.time = time
        self.acc = acc

def parse_line(line, version):
    if version == "v1":
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
    else:
        return None
    return LineResult(line_type, epoch, num_backprop, loss, time, acc)

def parse_file(filename):
    #print filename

    train_losses = []
    times = []
    accuracies = []
    num_backprops = []
    last_num_backprop = -1
    first_time = -1

    test_times = []
    test_num_backprops = []
    test_losses = []
    test_accuracies = []
    first_test_time = -1


    x = 0
    with open(filename) as f:
        for line in f:
            version = filename.split('.')[-1]
            line_result = parse_line(line, version)

            if line_result is None:
                continue

            if line_result.line_type == "train_debug":

                # Skip duplicate lines with same num_backprop
                if last_num_backprop == line_result.num_backprop:
                    continue
                last_num_backprop = line_result.num_backprop

                # Calculate time elapsed using first time recorded as start
                if first_time == -1:
                    first_time = line_result.time
                time_elapsed = line_result.time - first_time

                # Add values
                times.append(time_elapsed)
                train_losses.append(line_result.loss)
                num_backprops.append(line_result.num_backprop)
                accuracies.append(line_result.acc)

            elif line_result.line_type == "test_debug":

                # Calculate time elapsed using first time recorded as start
                if first_test_time == -1:
                    first_test_time = line_result.time
                time_elapsed = line_result.time - first_test_time

                # Add values
                test_times.append(time_elapsed)
                test_losses.append(line_result.loss)
                test_accuracies.append(line_result.acc)
                test_num_backprops.append(line_result.num_backprop)

    return num_backprops, \
            test_num_backprops, \
            times, \
            test_times, \
            train_losses, \
            test_losses, \
            accuracies, \
            test_accuracies

def plot(ys_by_config, xlabel, ylabel, plot_dir, xs_by_config):
    for config, ys in sorted(ys_by_config.iteritems(), key=lambda x: x[0].top_k):
        xs = xs_by_config[config]
        label = "top_{}/{}, lr={}".format(config.top_k, config.pool_size, config.lr)
        print xlabel, ylabel, len(xs), len(ys)
        plt.plot(xs, ys, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plot_file = "{}_{}_{}_top{}_ps{}_lr{}.pdf".format(config.dataset,
                                                      ylabel,
                                                      config.network,
                                                      config.top_k,
                                                      config.pool_size,
                                                      config.lr)

    plt.legend()
    plt.savefig(os.path.join(plot_dir, plot_file))
    plt.clf()

def plot_scalar(experiment_dir, plot_dir):
    num_backprops_by_config = {}
    test_num_backprops_by_config = {}
    train_losses_by_config = {}
    test_losses_by_config = {}
    times_by_config = {}
    test_times_by_config = {}
    accuracies_by_config = {}
    test_accuracies_by_config = {}

    for filename in os.listdir(experiment_dir):
        if filename == ".DS_Store":
            continue
        filepath = os.path.join(experiment_dir, filename)
        config = Config(filename)
        num_backprops, test_num_backprops, times, test_times, train_losses, test_losses, train_accuracies, test_accuracies \
                = parse_file(filepath)
        num_backprops_by_config[config] = num_backprops
        test_num_backprops_by_config[config] = test_num_backprops
        times_by_config[config] = times
        test_times_by_config[config] = test_times
        train_losses_by_config[config] = train_losses
        test_losses_by_config[config] = test_losses
        accuracies_by_config[config] = train_accuracies
        test_accuracies_by_config[config] = test_accuracies

    plot(train_losses_by_config, "Num Images Trained", "Training Loss", plot_dir, xs_by_config=num_backprops_by_config)
    plot(test_accuracies_by_config, "Num Images Trained", "Test Accuracy", plot_dir, xs_by_config=test_num_backprops_by_config)
    plot(test_losses_by_config, "Num Images Trained", "Test Loss", plot_dir, xs_by_config=test_num_backprops_by_config)
    #plot(train_losses_by_config, "Wall Clock Time", "Training Loss By Time", plot_dir, xs_by_config=times_by_config)
    #plot(accuracies_by_config, "Wall Clock Time", "Test Accuracy By Time", plot_dir, xs_by_config=test_times_by_config)
    #plot(test_losses_by_config, "Wall Clock Time", "Test Loss By Time", plot_dir, xs_by_config=test_times_by_config)

def main():
    plot_home_dir = "plots"

    experiment_name = "180915_mnist"
    experiment_dir = "data/output/mnist/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_scalar(experiment_dir, plot_dir)

    experiment_name = "180915_cifar10_mobilenet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #plot_scalar(experiment_dir, plot_dir)

    experiment_name = "180915_cifar10_resnet"
    experiment_dir = "data/output/cifar10/{}".format(experiment_name)
    plot_dir = "{}/{}".format(plot_home_dir, experiment_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_scalar(experiment_dir, plot_dir)

if __name__ == "__main__":
    main()


