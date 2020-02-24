
import os
import numpy as np
import pickle
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

def format_plot(xlabel, ylabel, label_size=10, grid=False):
    plt.tick_params(axis='y', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='y', which='minor', labelsize=label_size * 1.2)
    plt.tick_params(axis='x', which='major', labelsize=label_size * 1.4)
    plt.tick_params(axis='x', which='minor', labelsize=label_size * 1.2)
    plt.xlabel(xlabel, fontsize=label_size * 1.6)
    plt.ylabel(ylabel, fontsize=label_size * 1.6)
    plt.tight_layout()
    leg = plt.legend(loc=0, prop={'size': label_size * 1.5})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

def plot(nofilter_file, sb_file, plot_file):
    nofilter_epochs = []
    nofilter_backpropped = []
    nofilter_losses = []
    nofilter_accs = []

    sb_epochs = []
    sb_backpropped = []
    sb_losses = []
    sb_accs = []

    with open(nofilter_file) as f:
        for line in f:
            if "start" in line:
                continue
            elif "backward_debug" in line:
                vals = line.rstrip().split(",")
                num_backpropped = int(vals[4])
                nofilter_backpropped.append(num_backpropped)
            elif "eval_debug" in line:
                vals = line.rstrip().split(",")
                num_epochs = int(vals[1])
                loss = float(vals[2])
                acc = float(vals[3])
                nofilter_epochs.append(num_epochs)
                nofilter_losses.append(loss)
                nofilter_accs.append(acc)
            else:
                continue

    last_time_elapsed = 0
    with open(sb_file) as f:
        for line in f:
            if "start" in line:
                continue
            elif "backward_debug" in line:
                vals = line.rstrip().split(",")
                num_backpropped = int(vals[4])
                sb_backpropped.append(num_backpropped)
            elif "eval_debug" in line:
                vals = line.rstrip().split(",")
                num_epochs = int(vals[1])
                loss = float(vals[2])
                acc = float(vals[3])
                sb_epochs.append(num_epochs)
                sb_losses.append(loss)
                sb_accs.append(acc)
            else:
                continue

    l = min(len(sb_backpropped), len(sb_losses))
    l2 = min(len(nofilter_losses), len(nofilter_backpropped))
    print(l, l2)
    if l == 0 or l2 == 0:
        return

    plt.plot(sb_backpropped[:l], sb_losses[:l], label="SB")
    plt.plot(nofilter_backpropped[:l2], nofilter_losses[:l2], label="NoFilter")
    plt.legend()
    plt.ylim(0, max(max(sb_losses[:l]) + 0.1, max(nofilter_losses[:l]) + 0.1))
    format_plot("Num Backpropped", "Val Loss")
    plot_filename = "{}-loss.png".format(plot_file)
    #plt.savefig(plot_filename)
    plt.clf()

    plt.plot(sb_backpropped[:l], sb_accs[:l], label="SB")
    plt.plot(nofilter_backpropped[:l2], nofilter_accs[:l2], label="NoFilter")
    plt.legend()
    plt.ylim(0, 1)
    format_plot("Num Backpropped", "Val Acc")
    plot_filename = "{}-acc.png".format(plot_file)
    plt.savefig(plot_filename)
    plt.clf()

def analyze_losses(fnames, plot_labels):
    for fname, plot_label in zip(fnames, plot_labels):
        losses_by_epoch = {}
        with open(fname) as f:
            for line in f:
                vals = line.rstrip().split(",")
                epoch = int(vals[0])
                loss = float(vals[1])
                if epoch not in losses_by_epoch.keys():
                    losses_by_epoch[epoch] = []
                losses_by_epoch[epoch].append(loss)

        xs = []
        ys = []
        errs = []
        for epoch, losses in losses_by_epoch.iteritems():
            xs.append(epoch)
            ys.append(np.average(losses))
            errs.append(np.std(losses))

        plt.errorbar(xs, ys, yerr=errs, label=plot_label, alpha=0.5)
    plt.legend()
    format_plot("Epoch", "Average loss", label_size=15)
    plotfile = "plots/losses.png"
    plt.savefig(plotfile)
    plt.clf()

def plot_pickle(pickle_file, plot_file):
    dataset_size = 25000
    with open(pickle_file, "rb") as f:
        d = pickle.load(f)
    skips = []
    max_skipped = max(d.values())
    for i in range(dataset_size):
        if i in d.keys():
            skipped = d[i]
        else:
            skipped = 0
        skips.append(skipped)

    outfile = "{}-ranked.txt".format(plot_file)
    with open(outfile, "w+") as f:
        zipped = zip(skips, range(dataset_size))
        z = [(x, y) for x, y in sorted(zipped)]
        for x, y, in z:
            line = "example {}, skipped {}\n".format(y, x)
            f.write(line)


    plt.clf()
    plt.hist(skips, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.2)
    format_plot("Dist", "Number of skips", label_size=15) 
    plot_filename = "{}-dist.png".format(plot_file)
    plt.savefig(plot_filename)



if __name__ == "__main__":

    #expnames = ["200212-imdb-primed", "200212-imdb-ids", "200212-imdb-rev"]
    #expnames = ["200217-imdb-spline-acc-random", "200217-imdb-spline-gpy", "200217-imdb-spline", "200217-spline-loss-random", "200217-spline-loss-gpy"]
    expnames = ["200223-acc-gpy"]
    for expname in expnames:
        expdir = "data/{}/".format(expname)
        nfexpdir = "data/{}/".format("200212-imdb-parabola")
        plotdir = "/Users/angela/src/private/bias-in-datasets/active_learning/plots/{}".format(expname)
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        primes = [1, 1.0, 2.0, 3.0]
        betas = [1, .05, .1, .2, .3, .4, .8, 1.5]
        trials = range(120)
        seeds = [4444]

        print("Plotting to {}".format(plotdir))
        for seed in seeds:
            nofilter_file = os.path.join(nfexpdir, "nofilter-seed4445.txt")
            print("Baseline {}".format(nofilter_file))
            for prime in primes:
                for beta in betas:
                    for trial in trials:
                        filename = "sb-{}prime-{}beta-{}seed-{}trial.txt".format(prime, beta, seed, trial)
                        sb_file = os.path.join(expdir, filename)
                        if not os.path.exists(sb_file):
                            continue
                        print("Analyzing {}".format(sb_file))
                        plot_file = os.path.join(plotdir, filename)
                        plot(nofilter_file, sb_file, plot_file)

                        picklename = "sb-{}prime-{}beta-{}seed-{}trial.pickle".format(prime, beta, seed, trial)
                        pickle_file = os.path.join(expdir, picklename)
                        if not os.path.exists(pickle_file):
                            continue
                        #plot_pickle(pickle_file, plot_file)

