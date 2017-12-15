
import cPickle
import glob
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = cPickle.load(fo)
    return data_dict

def load_cifar10(dataset_path):
    data = []
    for path in sorted(glob.glob(dataset_path + '/*_batch*')):
        cifar10_data_batch = unpickle(path)
        data.append(cifar10_data_batch['data'])
    data = np.vstack(data)
    data_r = data[:, :1024].reshape(data.shape[0], 32, 32)
    data_g = data[:, 1024:2048].reshape(data.shape[0], 32, 32)
    data_b = data[:, 2048:].reshape(data.shape[0], 32, 32)
    data = np.stack([data_r, data_g, data_b], axis=-1)
    return data

