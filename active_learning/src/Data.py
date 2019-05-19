import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import metrics
import parser

class ParsedData:
    def __init__(self, filepath, experiment_name=None, filename=None):
        self.filepath = filepath
        self.train_lines, self.test_lines = parser.parse_file(filepath)

    @property
    def train_num_backwards(self):
        return [l.num_backprop / 1000000. for l in self.train_lines]

    @property
    def test_num_backwards(self):
        return [l.num_backprop / 1000000. for l in self.test_lines]

    @property
    def test_num_inferences(self):
        return [(l.num_backprop + l.num_skip) / 1000000. for l in self.test_lines]

    @property
    def train_losses(self):
        return [l.loss for l in self.train_lines]

    @property
    def test_losses(self):
        return [l.loss for l in self.test_lines]

    @property
    def train_accuracies(self):
        return [l.acc for l in self.train_lines]

    @property
    def test_accuracies(self):
        return [l.acc for l in self.test_lines]

    @property
    def test_errors(self):
        return [100 - l.acc for l in self.test_lines]

    @property
    def ratio_backpropped_ys(self):
        return [l.num_backprop / float(l.num_backprop + l.num_skip) \
                                           for l in self.test_lines \
                                           if (l.num_backprop + l.num_skip) > 0]
    @property
    def ratio_backpropped_xs_millions(self):
        return [l.num_backprop / 1000000. \
                    for l in self.test_lines \
                    if (l.num_backprop + l.num_skip) > 0]

    @property
    def final_accuracy(self):
        return self.test_accuracies[-1]

    def error_at_num_backwards(self, threshold):
        np_list = np.array(self.test_num_backwards)
        index = (np_list >= threshold).argmax() if (np_list >= threshold).any() else -1
        if index >= 0:
            return self.test_errors[index]
        else:
            return None

    def inferences_at_num_backwards(self, threshold):
        np_list = np.array(self.test_num_backwards)
        index = (np_list >= threshold).argmax() if (np_list >= threshold).any() else -1
        if index >= 0:
            return self.test_num_inferences[index]
        else:
            return None

    def auc(self, xmax = None):
        if xmax:
            filtered = [(x,y) for x, y in zip(self.ratio_backpropped_xs_millions,
                                              self.test_accuracies)
                        if x <= xmax] 
            filtered_xs = [d[0] for d in filtered]
            filtered_ys = [d[1] for d in filtered]
            return metrics.auc(filtered_xs,
                               filtered_ys)
        else:
            return metrics.auc(self.ratio_backpropped_xs_millions,
                               self.test_accuracies)

        
