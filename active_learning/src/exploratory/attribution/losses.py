import os
import numpy as np
import math
import matplotlib.pyplot as plt
import ast

import ast
import ast
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(xs, ys, zs, xlabel="x", ylabel="y", zlabel="z"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #ax.plot3D(xs, ys, zs, 'gray')
    ax.scatter3D(ys, xs, zs, c=zs, cmap='RdYlGn');

    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.set_zlabel(zlabel)

    plt.show()

def update_graph(num, xs, ys, zs):
    t = np.array([np.ones(100)*i for i in range(len(xs))]).flatten()
    df = pd.DataFrame({"time": t ,"x" : xs[:,0], "y" : ys[:,1], "z" : zs[:,2]})
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, time={}'.format(num))

def animate_3d(xs, ys, zs, xlabel="x", ylabel="y", zlabel="z"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(ys, xs, zs, c=zs, cmap='RdYlGn');
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.set_zlabel(zlabel)

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=40, blit=False)

    plt.show()

def new_loss(class_prob, other_softmaxs):
    other_max = max(other_softmaxs)
    a = math.log(class_prob)
    b = math.log(other_max)
    return -math.log(class_prob) - math.log(1 - (other_max - (1. - class_prob) / 9))
    #return a + 0.01 * b

def process(inputs, labels):
    new_losses = []
    empiricals = []
    for input_file, label in zip(inputs, labels):
        xs = []
        ys = []
        with open(input_file) as f:
            for line in f:
                if "train_debug" in line:
                    vals = line.split(",")
                    backprops = int(vals[2])
                    forwards = int(vals[2]) + int(vals[3])
                    loss = float(vals[5])
                    xs.append(backprops)
                    ys.append(loss)
        plt.plot(xs, ys, label=label)

    xs = []
    ys = []
    input_file = inputs[2]
    with open(input_file) as f:
        for line in f:
            if "train_debug" in line:
                vals = line.split(",")
                backprops = int(vals[2])
                loss = float(vals[5]) * 9
                xs.append(backprops)
                ys.append(loss)
    #plt.plot(xs, ys, label="Squared Transformed")
    plt.ylim(0, 2)
    plt.ylabel("Loss")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    inputs = []
    labels = []
    home = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/"
    filename = "sampling_cifar10_mobilenetv2_1_128_0.0_0.0005_trial1_seed1337_v2"
    filename2 = "sampling_cifar10_mobilenetv2_0.1_128_0.0_0.0005_trial1_seed1337_v2"

    a1 = os.path.join(home, "190414_forwardlr_cross_custom", filename)
    inputs.append(a1)
    labels.append("Baseline")

    a2 = os.path.join(home, "190414_forwardlr_cross_custom", filename2)
    inputs.append(a2)
    labels.append("SB w/ forward-pass based LR")

    #b = os.path.join(home, "190414_forwardlr_cross_squared", filename)
    #inputs.append(b)
    #labels.append("Cross Squared")

    c = os.path.join(home, "190414_forwardlr_probnormed2_cross_reweighted", filename)
    inputs.append(c)
    labels.append("XEntropy Reweighted Probnormed2")

    c = os.path.join(home, "190414_forwardlr_probnormed3_cross_reweighted", filename)
    #inputs.append(c)
    #labels.append("Cross Reweighted Probnormed3")

    c = os.path.join(home, "190205_sample", filename2)
    inputs.append(c)
    labels.append("SB")

    process(inputs, labels)
