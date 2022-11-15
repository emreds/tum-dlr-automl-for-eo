import numpy as np

def positive_persistence(acc_list):
    positive_persistence_over_time = [1.0]
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]

    previous_common_models = [True for i in range(len(previous_top25))]

    for epoch in range(1, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        # check the consecutive two epochs, find common models
        current_common_models = current_top25 == previous_top25

        # compare the current common models with previous commom models
        current_common_models = (previous_common_models == True) & (current_common_models == True) & (
                previous_common_models == current_common_models)

        positive_persistence_over_time.append(current_common_models.count(True)/float(top25_len))

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return positive_persistence_over_time


def negative_persistence(acc_list):
    negative_persistence_over_time = [1.0]
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[:top25_len]

    previous_common_models = [True for i in range(len(previous_top25))]

    for epoch in range(1, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[:top25_len]
        current_common_models = current_top25 == previous_top25

        current_common_models = (previous_common_models == True) & (current_common_models == True) & (
                previous_common_models == current_common_models)

        negative_persistence_over_time.append(current_common_models.count(True)/float(top25_len))

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return negative_persistence_over_time

def positive_persistence_auc(acc_list):
    positive_persistence_over_time = positive_persistence(acc_list)
    return sum(positive_persistence_over_time) / float(len(positive_persistence_over_time))
def negative_persistence_auc(acc_list):
    negative_persistence_over_time = negative_persistence(acc_list)
    return sum(negative_persistence_over_time)/float(len(negative_persistence_over_time))

def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))

