
import os
import sys
import numpy as np
from Data import ParsedData

class DataToTarget:
    def __init__(self, baseline_data, new_data, target_error):

        baseline_num_backprops = find_first_x_at_y_err(baseline_data.test_num_backprops,
                                                       baseline_data.test_errors,
                                                       target_error)
        new_num_backprops = find_first_x_at_y_err(new_data.test_num_backprops,
                                                  new_data.test_errors,
                                                  target_error)
        baseline_num_forwards = find_first_x_at_y_err(baseline_data.test_num_inferences,
                                                      baseline_data.test_errors,
                                                      target_error)
        new_num_forwards = find_first_x_at_y_err(new_data.test_num_inferences,
                                                 new_data.test_errors,
                                                 target_error)
        self.baseline_num_backprops = baseline_num_backprops
        self.baseline_num_forwards = baseline_num_forwards
        self.new_num_backprops = new_num_backprops
        self.new_num_forwards = new_num_forwards

    @property
    def percent_fewer_backwards(self):
        return (self.baseline_num_backprops - self.new_num_backprops) / float(self.baseline_num_backprops) * 100

    @property
    def percent_more_forwards(self):
        return (self.new_num_forwards - self.baseline_num_forwards) / float(self.baseline_num_forwards) * 100

def find_first_x_at_y_err(xs, ys, ymarker):
    if ymarker is None:
        return None
    for x, y in zip(xs, ys):
        if y <= ymarker:
            return x
    return None

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

def evaluate_multiverse(experiments_dir, baseline_name, baseline_file, experiment_name):

    baseline_data = ParsedData(os.path.join(experiments_dir, baseline_name, baseline_file),
                               baseline_name,
                               baseline_file)
    baseline_accuracy = baseline_data.final_accuracy
    exp_dir = os.path.join(experiments_dir, experiment_name)

    for experiment_filename in os.listdir(exp_dir):
        filepath = os.path.join(exp_dir, experiment_filename)
        if experiment_filename == baseline_file or filename == ".DS_store" or os.path.isdir(filepath):
            continue
        exp_data = ParsedData(filepath, experiment_name, experiment_filename)
        auc_diff = get_auc_diff(baseline_data, exp_data)
        print("AUC Difference: {}".format(auc_diff))

def evaluate(experiments_dir, baseline_name, baseline_file, experiment_name, max_error=20):
    baseline_data = ParsedData(os.path.join(experiments_dir, baseline_name, baseline_file),
                               baseline_name,
                               baseline_file)
    baseline_accuracy = baseline_data.final_accuracy
    baseline_error = 100 - baseline_accuracy
    exp_dir = os.path.join(experiments_dir, experiment_name)
    print("==============================================")
    print("Baseline Error: {}".format(baseline_error))

    for experiment_filename in os.listdir(exp_dir):
        filepath = os.path.join(exp_dir, experiment_filename)
        if experiment_filename == baseline_file or experiment_filename == "sha" or os.path.isdir(filepath):
            continue
        exp_data = ParsedData(filepath, experiment_name, experiment_filename)
        final_accuracy = exp_data.final_accuracy
        final_error = 100 - final_accuracy
        auc_diff = get_auc_diff(baseline_data, exp_data)

        print("-----------------------------------------------")
        percent_lower_error = (baseline_error - final_error) * 100. / baseline_error
        print("Final Error: {0:.3f}, {1:.3f}% lower".format(final_error, percent_lower_error))

        if baseline_error >= final_error:
            data_to_target = DataToTarget(baseline_data, exp_data, baseline_error)
            print("{0:.2f}% fewer backwards and {1:.2f}% more forwards to {2:.1f}% error".format(data_to_target.percent_fewer_backwards,
                                                                                                 data_to_target.percent_more_forwards,
                                                                                                 baseline_error))
            print("New: {0:.2f} million backwards, {1:.2f} million forwards".format(data_to_target.new_num_backprops,
                                                                                    data_to_target.new_num_forwards))
            print("Baseline: {0:.2f} million backwards, {1:.2f} million forwards".format(data_to_target.baseline_num_backprops,
                                                                                         data_to_target.baseline_num_forwards))
        else:
            print("SB increases error")


def evaluate_file(filename, target_errors=None):
    exp_data = ParsedData(filename)
    final_error = 100 - exp_data.final_accuracy
    print("Final error: {}".format(final_error))

    if target_errors:
        for target_error in target_errors:
            num_backprops = find_first_x_at_y_err(exp_data.test_num_backprops,
                                                  exp_data.test_errors,
                                                  target_error)
            num_forwards = find_first_x_at_y_err(exp_data.test_num_inferences,
                                                 exp_data.test_errors,
                                                 target_error)
            print "{},{},{}".format(target_error, num_backprops, num_forwards)
