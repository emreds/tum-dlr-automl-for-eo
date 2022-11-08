import numpy as np


def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))
