
import os
import sys
import numpy as np
from Data import ParsedData


def find_first_x_at_y(xs, ys, ymarker):
    if ymarker is None:
        return None
    for x, y in zip(xs, ys):
        if y >= ymarker:
            return x
    return None

def get_auc_diff(data1, data2):
    xmax = min(max(data1.test_num_backprops),
               max(data2.test_num_backprops))
    auc1 = data1.auc(xmax=xmax)
    auc2 = data2.auc(xmax=xmax)
    return auc2 - auc1

def get_X_more_backwards(data1, data2):
    target_accuracies = range(10, 100, 10) + range(91, 100)
    xs1 = [find_first_x_at_y(data1.test_num_backprops,
                             data1.test_accuracies,
                             ymarker)
            for ymarker in target_accuracies]
    xs2 = [find_first_x_at_y(data2.test_num_backprops,
                             data2.test_accuracies,
                             ymarker)
            for ymarker in target_accuracies]

    nums = [x2 for x1, x2 in zip(xs1, xs2) if x1 > 0 and x2 > 0]
    denoms = [x1 for x1, x2 in zip(xs1, xs2) if x1 > 0 and x2 > 0]
    percent_diffs = [n / float(d) for n, d in zip(nums, denoms)]
    return np.average(percent_diffs)


def get_X_more_forwards(data1, data2):
    target_accuracies = range(10, 100, 10) + range(91, 100)
    xs1 = [find_first_x_at_y(data1.test_num_inferences,
                              data1.test_accuracies,
                              ymarker)
            for ymarker in target_accuracies]
    xs2 = [find_first_x_at_y(data2.test_num_inferences,
                              data2.test_accuracies,
                              ymarker)
            for ymarker in target_accuracies]

    nums = [x2 for x1, x2 in zip(xs1, xs2) if x1 > 0 and x2 > 0]
    denoms = [x1 for x1, x2 in zip(xs1, xs2) if x1 > 0 and x2 > 0]
    percent_diffs = [n / float(d) for n, d in zip(nums, denoms)]
    return np.average(percent_diffs)

def evaluate_multiverse(experiments_dir, baseline_name, baseline_file, experiment_name):

    baseline_data = ParsedData(os.path.join(experiments_dir, baseline_name, baseline_file),
                               baseline_name,
                               baseline_file)
    baseline_accuracy = baseline_data.final_accuracy
    exp_dir = os.path.join(experiments_dir, experiment_name)

    for experiment_filename in os.listdir(exp_dir):
        filepath = os.path.join(exp_dir, experiment_filename)
        if experiment_filename == baseline_file or os.path.isdir(filepath):
            continue
        exp_data = ParsedData(filepath, experiment_name, experiment_filename)
        auc_diff = get_auc_diff(baseline_data, exp_data)
        print("AUC Difference: {}".format(auc_diff))

def evaluate(experiments_dir, baseline_name, baseline_file, experiment_name):
    baseline_data = ParsedData(os.path.join(experiments_dir, baseline_name, baseline_file),
                               baseline_name,
                               baseline_file)
    baseline_accuracy = baseline_data.final_accuracy
    exp_dir = os.path.join(experiments_dir, experiment_name)
    print("Baseline Accuracy: {}".format(baseline_accuracy))

    for experiment_filename in os.listdir(exp_dir):
        filepath = os.path.join(exp_dir, experiment_filename)
        if experiment_filename == baseline_file or experiment_filename == "sha" or os.path.isdir(filepath):
            continue
        exp_data = ParsedData(filepath, experiment_name, experiment_filename)
        final_accuracy = exp_data.final_accuracy
        auc_diff = get_auc_diff(baseline_data, exp_data)
        avg_X_more_backwards = get_X_more_backwards(baseline_data, exp_data)
        avg_X_more_forwards = get_X_more_forwards(baseline_data, exp_data)

        print("Final Accuracy: {}".format(final_accuracy))
        print("AUC Difference: {}".format(auc_diff))
        print("{0:.2f}X More Backprops".format(avg_X_more_backwards))
        print("{0:.2f}X More Forward Props".format(avg_X_more_forwards))
