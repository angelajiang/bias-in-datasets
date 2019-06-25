import math
import numpy as np
import pickle
import pprint as pp
import os
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF


def predict_iterative(pickle_prefix, fout):

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

    pickle_file = pickle_prefix + "probabilities.pickle"
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    selects_pickle_file = pickle_prefix + "selects.pickle"

    ys_by_image_id = {}
    selects_by_image_id = {}

    print("Loading probabilities...")
    with open(pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, ys in d.iteritems():
            ys_by_image_id[image_id] = ys

    print("Loading selects...")
    with open(selects_pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, selects in d.iteritems():
            selects_by_image_id[image_id] = selects

    max_ids = 10
    num_initial_training = 200
    mses = []

    for i, (image_id, ys) in enumerate(ys_by_image_id.iteritems()):
        if i > max_ids:
            break

        ys_train = ys[:num_initial_training]
        X = np.array(range(len(ys_train))).reshape(-1, 1)
        gp.fit(X, ys_train)
        selects = selects_by_image_id[image_id]
        print("-------------image-{}------------------".format(image_id))
        print("Fit image-{} with {} samples with {} average probability".format(image_id,
                                                                                len(ys_train),
                                                                                round(np.mean(ys_train), 4)))
        for i in range(num_initial_training+1, len(ys)):
            xs_test = np.array([i]).reshape(-1,1)
            y_pred, std = gp.predict(xs_test, return_std=True)
            print(std)
            is_selected = 1 - selects[i]
            fout.write("prediction,{},{},{},{}\n".format(image_id, (xs_test[0]).tolist()[0], y_pred[0], ys[i]))
            fout.flush()
            mse = get_mse(y_pred, [ys[i]])
            mses.append(mse)
            #print("MSE: {0:5f}".format(mse))

            ys_train = ys[:i]
            X = np.array(range(len(ys_train))).reshape(-1, 1)
            gp.fit(X, ys_train)
    return np.average(mses)

def get_mse(guesses, answers):
    squares = [math.pow(g - a, 2) for g, a in zip(guesses, answers)]
    return np.average(squares)

def get_nmse(guesses, answers):
    squares = [math.pow(g - a, 2) / float(np.average(a)) for g, a in zip(guesses, answers)]
    return np.average(squares)

if __name__ == "__main__":
    #exps = ["190617_probs_noaug_nosample", "190617_probs_aug_nosample", "190617_probs_noaug_sample", "190617_probs_aug_sample"]
    exps = ["190617_probs_aug_sample"]
    template = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/{}/pickles/probabilities_by_image/sampling_cifar10_mobilenetv2_0_128_1024_0.0005_trial1_seed1337_"
    output_dir = "outputs"

    output_exp = "190623_prediction"

    for exp in exps:
        print("================={}=====================".format(exp))
        output_subdir = os.path.join(output_dir, output_exp)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        output_file = "{}/{}".format(output_subdir, exp)
        with open(output_file, "w+") as fout:
            pickle_prefix = template.format(exp)
            average_mse = predict_iterative(pickle_prefix, fout=fout)
            fout.write("average_MSE,{}\n".format(average_mse))

    #predict(pickle_prefix, False)
