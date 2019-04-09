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

def process(input_file):
    xs = []
    ys = []
    new_losses = []
    empiricals = []
    with open(input_file) as f:
        for line in f:
            if "xentropy" in line:
                vals = line.split(";")
                empirical = float(vals[2])
                label = int(vals[3])

                softmax = ast.literal_eval(vals[1])
                softmax = [float(i) for i in softmax]

                class_prob = softmax[label]

                del softmax[label]
                max_loss = max(softmax)

                if class_prob >= 0.403 or class_prob < 0.4:
                    continue

                xs.append(class_prob)
                ys.append(max_loss)
                empiricals.append(empirical)
                new_losses.append(new_loss(class_prob, softmax))
            if len(xs) > 500:
                break

    #plot_3d(xs, ys, xentropys, "Class-Prediction", "Max-Prediction", "XEntropy")
    #plot_3d(xs, ys, new_losses, "Class-Prediction", "Max-Prediction", "New Losses")
    animate_3d(xs, ys, empiricals, "Class-Prediction", "Max-Prediction", "Empirical Cross Entropy")
    animate_3d(xs, ys, new_losses, "Class-Prediction", "Max-Prediction", "New Losses")

if __name__ == "__main__":
  input_file = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/loss_fn_exploration/out"
  process(input_file)
