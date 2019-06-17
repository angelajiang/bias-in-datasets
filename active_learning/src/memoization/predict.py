import pickle
import numpy as np
import pprint as pp
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def predict_regression(pickle_prefix):
    pickle_file = pickle_prefix + "probabilities.pickle"

    ys_by_image_id = {}

    with open(pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, ys in d.iteritems():
            ys_by_image_id[image_id] = ys
            #xs = range(len(ys))

    #kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    kernel = Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    max_ids = 2
    for i, (image_id, ys) in enumerate(ys_by_image_id.iteritems()):
        if i > max_ids:
            break
        X = np.array(range(len(ys))).reshape(-1, 1)
        gp.fit(X, ys)
        x_pred = np.linspace(len(ys), len(ys)+10).reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        print("================={}=====================".format(image_id))
        pp.pprint(zip([x[0] for x in x_pred], y_pred))


def predict_classification(pickle_prefix):
    pickle_file = pickle_prefix + "selects.pickle"

    ys_by_image_id = {}

    with open(pickle_file, 'rb') as handle:
        d = pickle.load(handle)
        for image_id, ys in d.iteritems():
            ys_by_image_id[image_id] = ys

    #kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    kernel = Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessClassifier(kernel=kernel)

    max_ids = 2
    for i, (image_id, ys) in enumerate(ys_by_image_id.iteritems()):
        if i > max_ids:
            break
        X = np.array(range(len(ys))).reshape(-1, 1)
        gp.fit(X, ys)
        x_pred = np.linspace(len(ys), len(ys)+10).reshape(-1,1)
        y_pred = gp.predict(x_pred)
        print("================={}=====================".format(image_id))
        pp.pprint(zip([x[0] for x in x_pred], y_pred))

if __name__ == "__main__":
    exp_name = "190617_probs_noaug_sample"
    template = "/Users/angela/src/private/bias-in-datasets/active_learning/data/output/cifar10/{}/pickles/probabilities_by_image/sampling_cifar10_mobilenetv2_0_128_1024_0.0005_trial1_seed1337_"

    pickle_prefix = template.format(exp_name)

    predict_regression(pickle_prefix)
    predict_classification(pickle_prefix)
