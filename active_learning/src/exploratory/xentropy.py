import numpy as np
import matplotlib.pyplot as plt
import ast

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(xs, ys, zs):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(xs, ys, zs, 50, cmap='binary')
    plt.show()

def process(input_file):
    xs = []
    ys = []
    zs = []
    with open(input_file) as f:
        for line in f:
            if "xentropy" in line:
                vals = line.split(";")
                softmax = ast.literal_eval(vals[1])
                xentropy = vals[2]
                softmax.sort()
                first = softmax[-1]
                second = softmax[-2]
                xs.append(first)
                ys.append(second)
                zs.append(xentropy)
            if len(xs) > 10000:
                break
    plot_3d(xs, ys, zs)

if __name__ == "__main__":
  input_file = "/Users/angela/src/projects/bias-in-datasets/active_learning/data/output/cifar10/loss_fn_exploration/out"
  process(input_file)
