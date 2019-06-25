import math
import numpy as np
import pickle
import pprint as pp
import os
from pathlib2 import Path
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF

def predict_selection(coin_flip, y_pred, std):
    if (y_pred - std) > coin_flip:
        prediction = 1
    else:
        prediction = 0
    #print("Y={} std={}, coin flip={}: Predicted={}".format(y_pred, std, coin_flip, prediction))
    return prediction

def load_data(pickle_file):
    parsed_pickle_file = "pickles/parsed_{}".format(os.path.basename(pickle_file))
    ppf = Path(parsed_pickle_file)
    if ppf.is_file():
        print("Loading {}...".format(parsed_pickle_file))
        with open(parsed_pickle_file, 'rb') as handle:
            vals_by_image_id = pickle.load(handle)
    else:
        print("Loading {}...".format(pickle_file))
        vals_by_image_id = {}
        with open(pickle_file, 'rb') as handle:
            d = pickle.load(handle)
            for image_id, ys in d.iteritems():
                vals_by_image_id[image_id] = ys
        with open(parsed_pickle_file, 'wb') as handle:
            pickle.dump(vals_by_image_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return vals_by_image_id

def predict_iterative(pickle_prefix, fout):

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    probabilities_pickle_file = pickle_prefix + "probabilities.pickle"
    selects_pickle_file = pickle_prefix + "selects.pickle"
    ys_by_image_id = load_data(probabilities_pickle_file)
    selects_by_image_id = load_data(selects_pickle_file)

    max_ids = 5
    num_initial_training = 200
    mses = []

    num_predictions = 0.
    num_correct = 0.
    num_positives = 0.
    num_negatives = 0.
    false_positives = 0.
    false_negatives = 0.

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
            is_selected_actual = 1 - selects[i]
            y_actual = ys[i]
            xs_test = np.array([i]).reshape(-1,1)
            coin_flip = np.random.uniform(0, 1)
            if coin_flip < y_actual:
                is_selected_hypothetical = 1
            else:
                is_selected_hypothetical = 0

            y_pred, std = gp.predict(xs_test, return_std=True)
            is_selected_pred = predict_selection(coin_flip, y_pred, std)

            fout.write("prediction,{},{},{},{},{},{},{}\n".format(image_id, (xs_test[0]).tolist()[0], y_pred[0], y_actual, is_selected_actual, is_selected_hypothetical, is_selected_pred))
            fout.flush()

            # Calculate accuracy of regression
            mse = get_mse(y_pred, [ys[i]])
            mses.append(mse)
            #print("MSE: {0:5f}".format(mse))

            # Calculate accuracy of selection
            num_predictions += 1
            if is_selected_pred == is_selected_actual:
                num_correct += 0
            if is_selected_actual:
                num_positives += 1
                if not is_selected_pred:
                    false_negatives += 1
            else:
                num_negatives += 1
                if not is_selected_pred:
                    false_positives += 1

            ys_train = ys[:i]
            X = np.array(range(len(ys_train))).reshape(-1, 1)
            gp.fit(X, ys_train)

        accuracy = num_correct / num_predictions
        false_positive_rate = false_positives /  num_negatives
        false_negative_rate = false_negatives /  num_positives
        print("Intermediate Acc:{}, FPR:{}, FNR:{}".format(accuracy,
                                                           false_positive_rate,
                                                           false_negative_rate))

    accuracy = num_correct / num_predictions
    false_positive_rate = false_positives /  num_negatives
    false_negative_rate = false_negatives /  num_positives
    return accuracy, false_positive_rate, false_negative_rate

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
            accuracy, false_positive_rate, false_negative_rate = predict_iterative(pickle_prefix, fout=fout)
            fout.write("accuracy,{},fpr,{},fnr,{}\n".format(accuracy,
                                                                                            false_positive_rate,
                                                                                            false_negative_rate))

    #predict(pickle_prefix, False)
