import numpy as np

def calculate_pairwise_dist(small_arch_array, hash_arch_array):
    ns, h, w = small_arch_array.shape
    n, h, w = hash_arch_array.shape
    flat_small_arch_array = small_arch_array.reshape(ns, h * w)
    flat_hash_arch_array = hash_arch_array.reshape(n, h * w)

    # Calculate the Hamming distance matrix using NumPy broadcasting
    # flat_small_arch_dim: 1000, 1, 289
    # flat_hash_arch_dim: 1, 423624, 289
    dist_matrix = np.bitwise_xor(
        flat_small_arch_array[:, np.newaxis, :], flat_hash_arch_array[np.newaxis, :, :]
    ).sum(axis=-1)

    return dist_matrix

def calculate_sample_pairwise_dist(one_arch, hash_arch_array):
    h, w = one_arch.shape
    n, h, w = hash_arch_array.shape
    flat_one_arch = one_arch.reshape(1, h * w)
    flat_hash_arch_array = hash_arch_array.reshape(n, h * w)

    # Calculate the Hamming distance matrix using NumPy broadcasting
    # flat_small_arch_dim: 1000, 1, 289
    # flat_hash_arch_dim: 1, 423624, 289
    dist_matrix = np.bitwise_xor(
        flat_one_arch[:, np.newaxis, :], flat_hash_arch_array[np.newaxis, :, :]
    ).sum(axis=-1)


    return dist_matrix