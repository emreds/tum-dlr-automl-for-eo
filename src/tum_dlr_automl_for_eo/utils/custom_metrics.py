import os
import numpy as np
import pandas as pd


def read_error_files(base_dir):
    base_dir = base_dir
    acc_list = []
    for file in os.listdir(base_dir):
        if "json" in file:
            json_path = os.path.join(base_dir, file)
            json_data = pd.read_json(json_path, lines=True)
            acc_list.append(json_data["train_acc"][0])

    acc_list = np.asarray(acc_list)
    return acc_list


def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))


def persistence(acc_list):
    num_of_models, num_of_epochs = np.shape(acc_list)
    top25_len = num_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]
    persistence = 1
    for epoch in range(1, num_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        common_num_of_elems = len(np.intersect1d(previous_top25, current_top25))
        persistence *= common_num_of_elems / num_of_models
        previous_top25 = current_top25

    return persistence
