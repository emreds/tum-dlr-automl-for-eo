import math
import os
from datetime import datetime

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
    return 50


def perform_bils_algorithm(architectures, acc_list, starting_point, num_of_nei_to_search, max_num_of_iter):
    """
    @return: list [found architecture index, found architecture's accuracy]
             in case of no improvement returns given architecture info.
    """

    curr_architecture_index = starting_point

    counter = 0

    while counter < max_num_of_iter:
        curr_architecture = architectures[curr_architecture_index]
        current_acc = acc_list[curr_architecture_index]
        found_arc_index = curr_architecture_index
        found_acc = current_acc

        neighbors = get_neighbors(curr_architecture, num_of_nei_to_search)

        # Loop over obtained 5 neighbors
        for neighbor in neighbors:
            found = False
            for arc_index, arc in enumerate(architectures):

                if all(neighbor == arc):
                    found = True
                    if acc_list[arc_index] > current_acc:
                        found_arc_index = arc_index
                        found_acc = acc_list[found_arc_index]
                    break
            if not found:
                accuracy = compute_architecture_accuracy(architecture=neighbor)
                # print('neighbor', np.expand_dims(neighbor,axis=0))
                # print('architectures', architectures)
                acc_list = np.append(acc_list,accuracy)
                architectures = np.append(architectures,np.expand_dims(neighbor,axis=0),axis=0)
                if accuracy > current_acc:
                    found_acc = accuracy
                    found_arc_index = len(architectures) - 1
                    # print(
                    #     f"from:{curr_architecture_index} to:{found_arc_index} pre_acc:{current_acc} found_acc:{found_acc}")

        if found_arc_index != curr_architecture_index:
            curr_architecture_index = found_arc_index
            counter += 1
        # if better architecture is not found
        else:
            break
    return found_arc_index, found_acc



def get_hamming_distance(architecture_1, architecture_2):
    return np.count_nonzero(architecture_1 != architecture_2)

"""
find starting points with high density of fully trained architectures, while simultaneously avoiding sampling twice
from the same area 
"""
def find_k_starting_points(architectures, k,radius=3):

    number_of_architectures = len(architectures)
    tabu_list = [0 for i in range(number_of_architectures)]

    indices = [np.arange(number_of_architectures)]
    n_of_architectures_nearby = [0 for i in range(number_of_architectures)]

    final_result = []
    for i in range(number_of_architectures):
        count_n_of_close_architectures = 0
        for j in range(number_of_architectures):
            distance = get_hamming_distance(architecture_1=architectures[j],architecture_2=architectures[i])
            if distance <= radius:
                count_n_of_close_architectures += 1
        n_of_architectures_nearby[i] = count_n_of_close_architectures

    sorted_indices = [x for _, x in sorted(zip(n_of_architectures_nearby, indices),reverse=True)]

    for i in range(number_of_architectures):
        if tabu_list[sorted_indices[i]] == 0 and len(final_result) < k:
            final_result.append(sorted_indices[i])
            for j in range(number_of_architectures):
                distance = get_hamming_distance(architecture_1=sorted_indices[i], architecture_2=architectures[j])
                if distance <= radius:
                    tabu_list[j] = 1

    while len(architectures) < k:
        sampled_index = sample(range(0, number_of_architectures), 1)[0]
        if sampled_index not in final_result:
            final_result.append(sampled_index)

    return final_result

def search_local_optima(architectures, acc_list, m_staring_points,
                        num_of_nei_to_search, max_num_of_iter_bills, number_of_iters):
    """
    Performs the BILS Algorithm for M Number of Starting Points
    Parameters
    ----------
    architectures: np.array [(Number of Models) x (length of binary encoded architecture)] binary encoded
    acc_list: np.array [Number of Models] Final Accuracies of Models @param starting_point
    m_staring_points: int Number of Starting Points
    num_of_nei_to_search: number of neighbors to search for each architecture
    max_num_of_iter: maximum number of iterations in BILS Algo.

    Returns Found: [[found architecture index, found architecture's accuracy]..., estimated number of local optima]
    -------

    """
    k_arr = []
    for i in range(number_of_iters):
        number_of_architectures = len(architectures)
        # track found architectures to decide number of local optima.
        k = 0
        # randomly choose M starting points
        starting_points = find_k_starting_points(architectures,m_staring_points)
        obtained_results = []
        for starting_point in starting_points:
            # perform BILS Algorithm for each starting points
            k += 1
            found_arch_index, found_acc = perform_bils_algorithm(architectures, acc_list, starting_point,
                                                                 num_of_nei_to_search, max_num_of_iter_bills)
            if [found_arch_index, found_acc] in obtained_results:
                k_arr.append(k)
                break
            else:
                obtained_results.append([found_arch_index, found_acc])
        if len(k_arr) != i+1 and k == len(starting_points):
            k_arr.append(k)

    k_mean = np.mean(np.array(k_arr))
    local_optima_estimation = math.pow(k_mean,2.0)/(-1*np.log(0.5))

    return k_arr, local_optima_estimation


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
