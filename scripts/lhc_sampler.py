from naslib.utils import nb101_api as api
from tum_dlr_automl_for_eo.utils import helper_functions
import pickle

# TODO 1: Give suitable paths to read nasbench_full.tfrecord file,
#         and nb101 dictionary and path to save the sampled_points
# TODO 2: Save the sampled points to use during local search and random walk
# TODO 3: Define number of architecture to sample

path_to_read_nb101_dict = ".../naslib/data/nb101_dict"
path_to_read_nb = '.../nasbench_full.tfrecord'
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
    with open(path_to_read_nb101_dict, 'rb') as f:
        nb101_dict = pickle.load(f)

    all_specs = []
    set_mats = set()
    while len(all_specs) < N:  # check if N samples found
        s = helper_functions.sample_single_valid_spec(helper_functions.NUM_VERTICES,
                                                      helper_functions.ALLOWED_OPS, nasbench)
        m, o, sp = s
        t_m = tuple(m.reshape(helper_functions.NUM_VERTICES * helper_functions.NUM_VERTICES))
        hash_m = nasbench._hash_spec(sp)
        adjacency_matrix = m.tolist()
        ops = helper_functions.rename_ops(o)
        encoded_sample = helper_functions.encode_matrix(adjacency_matrix, ops)
        sample_key = helper_functions.encoded_architecture_to_key(encoded_sample)

        if hash_m not in set_mats and sample_key in nb101_dict: # check if found key belongs to nb101
            set_mats = set_mats | set([t_m])
            # all_specs.append(s)
            all_specs.append(sample_key)
    return all_specs


def sample_architectures():
    nasbench = api.NASBench(path_to_read_nb,
                            data_format=data_format)

    sampled_architectures = LHS_sample_N_valid_specs(number_of_arc_to_sample, nasbench)

    filehandler = open(path_to_save_sampled_keys, 'wb')
    pickle.dump(sampled_architectures, filehandler)
    # TODO 4: read nb101 dictionary created by nb101_dict_creator,
    #         then train obtained architectures using created keys
    #         save this file to the same location again to use
    #         in local search


if __name__ == '__main__':
    sample_architectures()
