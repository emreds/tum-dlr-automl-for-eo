import numpy as np
from random import sample


def get_neighbors(architecture):
    """
    @param architecture: 1 binary encoded architecture np.array
    @return: 5 neighbors of given arc. with 1 Hamming Distance
    """
    # get 5 random indices to create 5 neighbors
    random_neighbor_indices = sample(range(0, len(architecture) - 1), 5)
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


def perform_bils_algorithm(architectures, acc_list, starting_point):
    """

    @param architectures: np.array [(Number of Models) x (length of binary encoded architecture)] binary encoded
    architectures
    @param acc_list: np.array [Number of Models] Final Accuracies of Models @param starting_point: int
    Starting model to search
    @return: list [found architecture index, found architecture's accuracy]
             in case of no improvement returns given architecture info.
    """
    MAX_NUMBER_OF_ITERATIONS = 50

    architectures = architectures
    curr_architecture_index = starting_point

    counter = 0

    while counter < MAX_NUMBER_OF_ITERATIONS:

        curr_architecture = architectures[curr_architecture_index]
        current_acc = acc_list[curr_architecture_index]
        found_arc_index = curr_architecture_index
        found_acc = current_acc

        neighbors = get_neighbors(curr_architecture)

        # Loop over obtained 5 neighbors
        for neighbor in neighbors:
            for arc_index, arc in enumerate(architectures):

                if all(neighbor == arc) and acc_list[arc_index] > current_acc:
                    found_arc_index = arc_index
                    found_acc = acc_list[found_arc_index]
                    # print(
                    #     f"from:{curr_architecture_index} to:{found_arc_index} pre_acc:{current_acc} found_acc:{found_acc}")

        if found_arc_index != curr_architecture_index and found_acc != current_acc:
            curr_architecture_index = found_arc_index
            counter += 1

        # if better architecture is not found
        else:
            break;
    return found_arc_index, found_acc


def search_local_optima(architectures, acc_list, m_staring_points):
    """
    Performs the BILS Algorithm for M Number of Starting Points
    @param architectures: np.array [(Number of Models) x (length of binary encoded architecture)] binary encoded
    architectures
    @param acc_list: np.array [Number of Models] Final Accuracies of Models @param starting_point: int
    Starting model to search
    @param m_staring_points: int Number of Starting Points
    @return: list [[Found Architecture Indices, Found Architectures' Accuracy]]
    """
    number_of_architectures = len(architectures)
    # randomly choose M starting points
    starting_points = sample(range(0, number_of_architectures - 1), m_staring_points)
    obtained_results = []
    for starting_point in starting_points:
        # perform BILS Algorithm for each starting points

        found_arch_index, found_acc = perform_bils_algorithm(architectures, acc_list, starting_point)
        if [found_arch_index, found_acc] not in obtained_results:
            obtained_results.append([found_arch_index, found_acc])

    return obtained_results









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
