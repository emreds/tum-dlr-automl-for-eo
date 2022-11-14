import numpy as np


def positive_persistence(acc_list):
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]
    current_top25 = np.argsort(acc_list[:, 1])[-top25_len:]
    # check the first two epochs, find common models
    initial_common_models = previous_top25 == current_top25

    if not any(initial_common_models):
        return 0

    previous_common_models = initial_common_models
    previous_top25 = current_top25

    for epoch in range(2, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        # check the consecutive two epochs, find common models
        current_common_models = current_top25 == previous_top25
        # compare the current common models with previous commom models
        current_common_models = (previous_common_models == True) & (current_common_models == True) & (
                previous_common_models == current_common_models)

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return len(np.where(current_common_models == True)[0]) / len(np.where(initial_common_models == True)[0])


def negative_persistence(acc_list):
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[:top25_len]
    current_top25 = np.argsort(acc_list[:, 1])[:top25_len]
    initial_common_models = previous_top25 == current_top25

    if not any(initial_common_models):
        return 0

    previous_common_models = initial_common_models
    previous_top25 = current_top25

    for epoch in range(2, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[:top25_len]
        current_common_models = current_top25 == previous_top25
        current_common_models = (previous_common_models == True) & (current_common_models == True) & (
                previous_common_models == current_common_models)

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return len(np.where(current_common_models == True)[0]) / len(np.where(initial_common_models == True)[0])

def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))

