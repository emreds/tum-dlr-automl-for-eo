from tum_dlr_automl_for_eo.utils import helper_functions
from naslib.utils import nb101_api as api
import pickle
import logging as log


# give the path to read nb101.tfrecord file, if not installed yet
# get it from https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

# TODO: Give suitable paths to read nasbench_full.tfrecord file and
#       path to save the nb101_dict file

path_to_read_data = '.../nasbench_full.tfrecord'
path_to_save_dictionary = ".../naslib/data/nb101_dict"

data_format = 'tfrecord'  # data format to read


def create_encoded_arc_file():
    nasbench = api.NASBench(path_to_read_data, data_format = data_format)
    encoded_to_architecture_dict = dict()
    for unique_has in nasbench.hash_iterator():
        architecture, _ = nasbench.get_metrics_from_hash(unique_has)
        encoded_architecture = helper_functions.encode_architecture(architecture)
        encoded_architecture_str = helper_functions.encoded_architecture_to_key(encoded_architecture)
        encoded_to_architecture_dict[encoded_architecture_str] = {
            'module_adjacency': architecture['module_adjacency'].tolist(),
            'module_operations': architecture['module_operations'],
            'accuracy': None # save accuracy as None to calculate later
        }
    filehandler = open(path_to_save_dictionary, 'wb')
    pickle.dump(encoded_to_architecture_dict, filehandler)



if __name__ == '__main__':
    create_encoded_arc_file()
