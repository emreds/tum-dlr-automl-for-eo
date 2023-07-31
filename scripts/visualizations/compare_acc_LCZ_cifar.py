import os
import matplotlib.pyplot as plt
import json
from pkgutil import iter_modules
import naslib
from naslib.utils import get_dataset_api, setup_logger, utils
from naslib.utils import nb101_api as api
import numpy as np


config = utils.get_config_from_args(config_type="nas")
dataset_api = get_dataset_api(config.search_space, config.dataset)

dir_path = '/home/strawberry/TUM/DLR/'

list_of_architectures = ['arch_0', 'arch_1']
arch_specs_file = '/home/strawberry/TUM/DLR/arch_specs.json'


def getAccuraciesLCZ(val=False):
    result_list = []

    for arch in list_of_architectures:
        filename = os.path.join(os.path.join(os.path.join(dir_path,arch),'version_0'),'metrics.csv')
        with open(filename) as file:
            lines = [line.rstrip() for line in file]

        validation_accuracy_index = 1
        training_accuracy_index = 6
        epoch_index = 3

        if val==True: # last line train accuracy, before that val
            last_line = lines[-2]
            last_line = last_line.split(',')
            result_list.append(float(last_line[validation_accuracy_index]))
        else:
            last_line = lines[-1]
            last_line = last_line.split(',')
            result_list.append(float(last_line[training_accuracy_index]))

    return result_list


def getCifarAccuracies(val=False):
    path_to_read_data = '/home/strawberry/TUM/DLR/tum-dlr-automl-for-eo/notebooks/src/naslib/naslib/data/nasbench_only108.pkl'  # 'nasbench_only108.tfrecord'
    data_format = 'tfrecord'

    nasbench = api.NASBench(path_to_read_data)

    result_list = []

    with open('/home/strawberry/TUM/DLR/arch_specs.json', 'rb') as f:
        arch_specs = json.load(f)
    for arch_spec_name in list_of_architectures:
        for i in range(len(arch_specs)):
            if arch_specs[i]['arch_code'] == arch_spec_name:
                matrix = arch_specs[i]['module_adjacency']
                ops = arch_specs[i]['module_operations']
                spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
                data = nasbench.query(spec)
                if val==True:
                    result_list.append(float(data['validation_accuracy']))
                else:
                    result_list.append(float(data['train_accuracy']))

    return result_list



val_accuraciesLCZ = getAccuraciesLCZ(True)
# print(val_accuraciesLCZ)
train_accuraciesLCZ = getAccuraciesLCZ(False)
# print(train_accuraciesLCZ)

val_accuraciesCifar = getCifarAccuracies(True)
# print(val_accuraciesCifar)
train_accuraciesCifar = getCifarAccuracies(False)

for i in range(len(val_accuraciesCifar)):
    print('Architecture',list_of_architectures[i],'statistics:')
    print('  - Train accuracy CIFAR10:%.2f' % train_accuraciesCifar[i])
    print('  - Train accuracy So2Sat LCZ42:%.2f' % train_accuraciesLCZ[i])
    print('  - Validation accuracy CIFAR10:%.2f' % val_accuraciesCifar[i])
    print('  - Validation accuracy So2Sat LCZ42:%.2f' % val_accuraciesLCZ[i])
