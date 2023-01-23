from tum_dlr_automl_for_eo.utils import helper_functions
from naslib.utils import nb101_api as api
from naslib.utils import get_dataset_api, utils
import pickle
import logging as log

# give the path to read nb101.pkl file, if not installed yet
# get it from https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

path_to_read_data = "../../nasbench_only108.pkl"
path_to_save_dictionary = "./nb101_dict"

data_format = 'pickle'  # data format to read


def create_encoded_arc_file():
    nasbench = api.NASBench(path_to_read_data, data_format = data_format)
    config = utils.get_config_from_args(config_type="nas")
    encoded_to_architecture_dict = dict()
    for unique_hash in nasbench.hash_iterator():
        architecture, _ = nasbench.get_metrics_from_hash(unique_hash)
        encoded_architecture = helper_functions.encode_architecture(architecture)
        encoded_architecture_str = helper_functions.encoded_architecture_to_key(encoded_architecture)
        encoded_to_architecture_dict[encoded_architecture_str] = {
            'module_adjacency': architecture['module_adjacency'].tolist(),
            'module_operations': architecture['module_operations'],
            'accuracy': None # save accuracy as None to calculate later
        }
    filehandler = open(path_to_save_dictionary, 'wb')
    print(len(encoded_to_architecture_dict.keys()))
    pickle.dump(encoded_to_architecture_dict, filehandler)


if __name__ == '__main__':
    create_encoded_arc_file()
"""
NASLIB RAW DICT STRUCTURE: 
ARCHITECTURE: {
    'module_adjacency': array([[0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0]],
       dtype=int8),
       
       'module_operations': ['input', 'maxpool3x3', 'conv3x3-bn-relu', 'maxpool3x3', 'conv1x1-bn-relu', 'maxpool3x3', 'output'],
       
       'trainable_parameters': 3468426}
       
EMPTY BLOCK: {108: [{
    'halfway_training_time': 702.2459716796875,
    'halfway_train_accuracy': 0.659254789352417,
    'halfway_validation_accuracy': 0.6333132982254028,
    'halfway_test_accuracy': 0.6277043223381042,
    'final_training_time': 1405.5989990234375,
    'final_train_accuracy': 0.9997996687889099,
    'final_validation_accuracy': 0.8970352411270142,
    'final_test_accuracy': 0.8939303159713745
    },
    {'halfway_training_time': 700.7080078125,
    'halfway_train_accuracy': 0.7763421535491943,
    'halfway_validation_accuracy': 0.7393830418586731,
    'halfway_test_accuracy': 0.7318710088729858,
    'final_training_time': 1402.673095703125,
    'final_train_accuracy': 1.0,
    'final_validation_accuracy': 0.9086538553237915,
    'final_test_accuracy': 0.90234375
    },
    {
    'halfway_training_time': 700.573974609375,
    'halfway_train_accuracy': 0.7804487347602844,
    'halfway_validation_accuracy': 0.7337740659713745,
    'halfway_test_accuracy': 0.7131410241127014,
    'final_training_time': 1401.5679931640625,
    'final_train_accuracy': 0.9997996687889099,
    'final_validation_accuracy': 0.8991386294364929,
    'final_test_accuracy': 0.8937299847602844}]
    }

UNIQUE_HASH: 00019e2cbab0bc93ca1a7816a099b2b

"""