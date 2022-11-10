import numpy as np


def positive_persistence(acc_list):
    pos_persistence = 1
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    pos_previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]

    for epoch in range(1, number_of_epochs):
        pos_current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        pos_common_num_of_models = np.sum(pos_current_top25 == pos_previous_top25)
        pos_persistence *= pos_common_num_of_models / number_of_models
        pos_previous_top25 = pos_current_top25

    return pos_persistence


def negative_persistence(acc_list):
    neg_persistence = 1
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    neg_previous_top25 = np.argsort(acc_list[:, 0])[:top25_len]

    for epoch in range(1, number_of_epochs):
        neg_current_top25 = np.argsort(acc_list[:, epoch])[:top25_len]
        neg_common_num_of_models = np.sum(neg_current_top25 == neg_previous_top25)
        neg_persistence *= neg_common_num_of_models / number_of_models
        neg_previous_top25 = neg_current_top25
    return neg_persistence
