from naslib.utils import nb101_api as api
from tum_dlr_automl_for_eo.utils import helper_functions
import pickle

# TODO 1: Give suitable paths to read nasbench_full.tfrecord file and
#       path to save the sampled_points
# TODO 2: Save the sampled points to use during local search and random walk
# TODO 3: Define number of architecture to sample


path_to_read_data = '.../nasbench_full.tfrecord'
path_to_save_sampled_keys = '.../naslib/data/sampled_points'

data_format = 'tfrecord'  # data format to read

number_of_arc_to_sample = 10


def LHS_sample_N_valid_specs(N, nasbench):
    """
    @param N: Number of Architecture to Sample
    @param nasbench: Nasbench(nas101) Object
    @return: 2D array [architecture_index][0] -> Adjacency Matrix
                      [architecture_index][1] -> Operations
    """
    all_specs = []
    set_mats = set()
    while len(set_mats) < N:
        s = helper_functions.sample_single_valid_spec(helper_functions.NUM_VERTICES,
                                                      helper_functions.ALLOWED_OPS, nasbench)
        m, o, sp = s
        t_m = tuple(m.reshape(helper_functions.NUM_VERTICES * helper_functions.NUM_VERTICES))
        hash_m = nasbench._hash_spec(sp)
        if hash_m not in set_mats:
            set_mats = set_mats | set([t_m])
            # all_specs.append(s)
            all_specs.append((m.tolist(), o))
    return all_specs


def sample_architectures():
    nasbench = api.NASBench(path_to_read_data,
                            data_format=data_format)

    sampled_architectures = LHS_sample_N_valid_specs(number_of_arc_to_sample, nasbench)

    architecture_keys = []
    for i in range(number_of_arc_to_sample):
        adjacency_matrix = sampled_architectures[i][0]
        ops = sampled_architectures[i][1]
        ops = helper_functions.rename_ops(ops)
        encoded_architecture = helper_functions.encode_matrix(adjacency_matrix, ops)
        architecture_key = helper_functions.encoded_architecture_to_key(encoded_architecture)
        architecture_keys.append(architecture_key)
    filehandler = open(path_to_save_sampled_keys, 'wb')
    pickle.dump(architecture_keys, filehandler)
    # TODO 4: read nb101 dictionary created by nb101_dict_creator,
    #         then train those architectures and after saving their accuracies to dictionary
    #         save this file to the same location again


if __name__ == '__main__':
    sample_architectures()
