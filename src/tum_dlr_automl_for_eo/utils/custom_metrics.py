import math
import os
import random
from datetime import datetime
from tum_dlr_automl_for_eo.utils.helper_functions import encoded_architecture_to_key, key_to_encoded_architecture

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from random import sample


def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))


def get_neighbors(architecture, num_of_nei_to_search):
    """
    @param num_of_nei_to_search: number of neighbors to search
    @param architecture: 1 binary encoded architecture np.array
    @return: neighbors (number of neighbors to search) of given arc. with 1 Hamming Distance
    """
    # get random neighbor indices to create (num_of_nei_to_search) neighbors
    random_neighbor_indices = sample(range(0, len(architecture)), num_of_nei_to_search)
    neighbors = []
    for index in random_neighbor_indices:
        # Change architecture to keep Hamming Distance at 1
        if architecture[index] == 0:
            curr_architecture = architecture.copy()
            curr_architecture[index] = 1
        else:
            curr_architecture = architecture.copy()
            curr_architecture[index] = 0
        neighbors.append(curr_architecture)
    return neighbors


# TODO change this to actually fitness estimation
def compute_architecture_accuracy(architecture):
    return random.random()


def perform_bils_algorithm(sampled_architectures, nb101_dict, starting_point,
                           num_of_nei_to_search, max_num_of_iter):
    """
    @return: list [found architecture index, found architecture's accuracy]
             in case of no improvement returns given architecture info.
    """
    # get the actual index from nb101 dict.
    starting_architecture_key = sampled_architectures[starting_point]
    curr_architecture_index = list(nb101_dict).index(starting_architecture_key)

    counter = 0

    while counter < max_num_of_iter:
        curr_architecture_key = list(nb101_dict.keys())[curr_architecture_index]  # key of current architecture

        # we will have the accuracy of the sampled architectures, for now assign a random value

        if (nb101_dict[curr_architecture_key]['accuracy']) is None:
            nb101_dict[curr_architecture_key]['accuracy'] = compute_architecture_accuracy(curr_architecture_key)

        current_acc = nb101_dict[curr_architecture_key]['accuracy']

        found_arc_index = curr_architecture_index
        found_acc = current_acc
        # arc: str -> list to get neighbours
        curr_architecture_encoded = key_to_encoded_architecture(curr_architecture_key)
        neighbors = get_neighbors(curr_architecture_encoded,
                                  num_of_nei_to_search)  # return binary neighbors as list

        # go through neighbors
        for neighbor in neighbors:

            # get the key of neighbor
            neighbor_key = encoded_architecture_to_key(neighbor)
            if neighbor_key in nb101_dict:  # if current neighbor in nb101, a valid architecture

                if nb101_dict[neighbor_key]['accuracy'] is None:  # if the performance of nei. not calculated yet
                    nb101_dict[neighbor_key]['accuracy'] = compute_architecture_accuracy(
                        nb101_dict[neighbor_key])

                if nb101_dict[neighbor_key]['accuracy'] > current_acc:  # if a better performance is found
                    found_arc_index = list(nb101_dict).index(neighbor_key)
                    found_acc = nb101_dict[neighbor_key]['accuracy']
                    # print(f"neighbor index: {found_arc_index} accuracy: {found_acc}")
            else:
                pass
                # print(f"neighbor is not valid !!!")
        if found_arc_index != curr_architecture_index:
            curr_architecture_index = found_arc_index
            counter += 1
        # if better architecture is not found
        else:
            break
    return found_arc_index, found_acc


def search_local_optima(sampled_architectures, nb101_dict, m_staring_points,
                        num_of_nei_to_search, max_num_of_iter_bills, number_of_iters):
    """
    Performs the BILS Algorithm for M Number of Starting Points
    Parameters
    ----------
    sampled_architectures: [str]  contains the keys of starting points, sampled from LHC
    nb101_dict: contains nas101 dictionary, {keys_of_architectures: module_adjacency,
                                                                    module_operations,
                                                                    accuracy}
    e.g: list(architectures.values())[3]
    {'module_adjacency': [[0, 1, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0]],
    'module_operations': ['input',
                               'conv1x1-bn-relu',
                               'conv3x3-bn-relu',
                               'maxpool3x3',
                               'conv1x1-bn-relu',
                               'maxpool3x3',
                               'output'],
    'accuracy': None}
    m_staring_points: int Number of Starting Points
    num_of_nei_to_search: int number of neighbors to search for each architecture
    max_num_of_iter_bills: int maximum number of iterations in BILS Algo.
    number_of_iters: int number of iterations for local search
    -------
    """
    k_arr = []
    for i in range(number_of_iters):
        number_of_architectures = len(sampled_architectures)
        # track found architectures to decide number of local optima.
        k = 0
        # randomly choose M starting points
        starting_points = sample(range(0, number_of_architectures), m_staring_points)
        obtained_results = []
        for starting_point in starting_points:
            k += 1
            found_arch_index, found_acc = perform_bils_algorithm(sampled_architectures, nb101_dict, starting_point,
                                                                 num_of_nei_to_search, max_num_of_iter_bills)
           # print(f"found architecture index: {found_arch_index} found accuracy: {found_acc}")
            if [found_arch_index, found_acc] in obtained_results:
                k_arr.append(k)
                break
            else:
                obtained_results.append([found_arch_index, found_acc])
    # k_mean = np.mean(np.array(k_arr))
    # local_optima_estimation = math.pow(k_mean, 2.0) / (-1 * np.log(0.5))
    return obtained_results


def positive_persistence(acc_list):
    positive_persistence_over_time = [1.0]
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]

    previous_common_models = np.array([True for i in range(len(previous_top25))])

    for epoch in range(1, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        # check the consecutive two epochs, find common models
        current_common_models = current_top25 == previous_top25

        # compare the current common models with previous commom models
        current_common_models = (previous_common_models == True) & (current_common_models == True) \
                                & (previous_common_models == current_common_models)

        positive_persistence_over_time.append(np.count_nonzero(current_common_models) / float(top25_len))

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return positive_persistence_over_time


def negative_persistence(acc_list):
    negative_persistence_over_time = [1.0]
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[:top25_len]

    previous_common_models = np.array([True for i in range(len(previous_top25))])

    for epoch in range(1, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[:top25_len]
        current_common_models = current_top25 == previous_top25

        current_common_models = (previous_common_models == True) & (current_common_models == True) & (
                previous_common_models == current_common_models)

        negative_persistence_over_time.append(np.count_nonzero(current_common_models) / float(top25_len))

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return negative_persistence_over_time


def positive_persistence_auc(acc_list):
    positive_persistence_over_time = positive_persistence(acc_list)
    return sum(positive_persistence_over_time) / float(len(positive_persistence_over_time))


def negative_persistence_auc(acc_list):
    negative_persistence_over_time = negative_persistence(acc_list)
    return sum(negative_persistence_over_time) / float(len(negative_persistence_over_time))


def ruggedness(data, lags=1):
    """
    Calculates ruggedness metric and returns the result.
    By definition calculates the Autocorrelation and takes the inverse of it.

    Args:
        data (np.array): Data to calculate ruggedness.
        lags (int): Number of lags to calculate.

    Returns:
        np.array
    """
    acorr = sm.tsa.acf(data, nlags=lags)
    rugs = 1 / acorr

    return rugs


def plot_ruggedness(data, exp_name, lags=1):
    """
    Plots the ruggedness.

    Args:
        data (np.array): Data to calculate autocorrelation.
        exp_name (string): Experiment name.
        lags (int): Number of lags to calculate.
    """
    rugs = ruggedness(data=data, lags=lags)
    exp_name += "_ruggedness"
    fig_name = get_fig_name(exp_name=exp_name, lags=lags)
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../reports/figures", fig_name)
    )
    plt.plot(rugs)
    plt.xticks(range(0, lags))
    plt.savefig(path)

    pass


def get_fig_name(exp_name, **kwargs):
    """
    Creates a unique figure name based on exp_name, parameters and datetime.

    Args:
        exp_name (string): Experiment name.

    Returns:
        string: Figure name.
    """

    fig_date = datetime.now().strftime("%Y%m%d%H%M%S")
    fig_name = f"{exp_name}_{fig_date}"
    if kwargs:
        for param, value in kwargs.items():
            fig_name += "_" + str(param) + "_" + str(value)

    fig_name += ".png"

    return fig_name
