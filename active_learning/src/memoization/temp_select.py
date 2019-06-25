import math
import numpy as np
import pickle
import pprint as pp
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF


def predict_iterative(pickle_prefix, is_regression, fout):

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

    pickle_file = pickle_prefix + "probabilities.pickle"
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    selects_pickle_file = pickle_prefix + "selects.pickle"

    ys_by_image_id = {}
    selects_by_image_id = {}

    print("Loading selects...")
    with open(selects_pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, selects in d.iteritems():
            selects_by_image_id[image_id] = selects

    print("Loading probabilities...")
    with open(pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, ys in d.iteritems():
            ys_by_image_id[image_id] = ys

    max_ids = 50
    num_initial_training = 0
    mses = []

    for i, (image_id, ys) in enumerate(ys_by_image_id.iteritems()):
        if i > max_ids:
            break
        print("-------------image-{}------------------".format(image_id))
        selects = selects_by_image_id[image_id]
        for i in range(len(ys)):
            xs_test = np.array([i]).reshape(-1,1)
            is_selected = 1 - selects[i]
            fout.write("prediction,{},{},{},{},{}\n".format(image_id, (xs_test[0]), 0, ys[i], is_selected))
            fout.flush()
    return 0

def get_mse(guesses, answers):
    squares = [math.pow(g - a, 2) for g, a in zip(guesses, answers)]
    return np.average(squares)

def get_nmse(guesses, answers):
    squares = [math.pow(g - a, 2) / float(np.average(a)) for g, a in zip(guesses, answers)]
    return np.average(squares)

if __name__ == "__main__":
    exps = ["190617_probs_noaug_sample", "190617_probs_aug_sample", "190617_probs_noaug_nosample", "190617_probs_aug_nosample"]
    template = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/{}/pickles/probabilities_by_image/sampling_cifar10_mobilenetv2_0_128_1024_0.0005_trial1_seed1337_"

    output_dir = "outputs"

    for exp in exps:
        print("================={}=====================".format(exp))
        output_file = "{}/{}_selects".format(output_dir, exp)
        with open(output_file, "w+") as fout:
            pickle_prefix = template.format(exp)
            average_mse = predict_iterative(pickle_prefix, is_regression=True, fout=fout)
            fout.write("average_MSE,{}\n".format(average_mse))

    #predict(pickle_prefix, False)
