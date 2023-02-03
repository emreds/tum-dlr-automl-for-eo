import os
import matplotlib.pyplot as plt
import json
from pkgutil import iter_modules
import naslib
from naslib.utils import get_dataset_api, setup_logger, utils
from naslib.utils import nb101_api as api
from tum_dlr_automl_for_eo.utils.custom_metrics import mean_accuracy, variance, positive_persistence_auc, positive_persistence, negative_persistence_auc, negative_persistence
import numpy as np


config = utils.get_config_from_args(config_type="nas")
dataset_api = get_dataset_api(config.search_space, config.dataset)

dir_path = '/home/strawberry/TUM/DLR/architecture_accuracies_So2Sat'


list_of_architectures = []#['arch_0', 'arch_1']
for i in range(9999):
    list_of_architectures.append('arch_'+str(i))

evaluated_list_of_architectures = []
arch_specs_file = '/home/strawberry/TUM/DLR/arch_specs.json'


def getAccuraciesLCZ(val=False, getLastOnly=True):
    result_list = [] # either [accuracy] or [accuracy, epoch]

    for arch in list_of_architectures:
        filename = os.path.join(os.path.join(os.path.join(dir_path,arch),'version_0'),'metrics.csv')
        try:
            with open(filename) as file:
                lines = [line.rstrip() for line in file]

            validation_accuracy_index = 1
            training_accuracy_index = 6
            epoch_index = 3

            if val==True: # last line train accuracy, before that val
                if getLastOnly:
                    last_line = lines[-2]
                    last_line = last_line.split(',')
                    result_list.append(float(last_line[validation_accuracy_index]))
                else:
                    for i in range(len(lines)):
                        if i==7 or i==25 or i==73 or i==215:
                            if i==7:
                                result_list.append([])
                            line = lines[i]
                            line = line.split(',')
                            result_list[-1].append(float(line[validation_accuracy_index]))
            else:
                last_line = lines[-1]
                last_line = last_line.split(',')
                result_list.append(float(last_line[training_accuracy_index]))
            if arch not in evaluated_list_of_architectures:
                evaluated_list_of_architectures.append(arch)
        except Exception as ex:
            pass
    return result_list


def getCifarAccuracies(val=False,getLastOnly=True):
    path_to_read_data = '/home/strawberry/TUM/DLR/tum-dlr-automl-for-eo/notebooks/src/naslib/naslib/data/nasbench_full.tfrecord'#'/home/strawberry/TUM/DLR/tum-dlr-automl-for-eo/notebooks/src/naslib/naslib/data/nasbench_full.tfrecord' #nasbench_only108.pkl'  # 'nasbench_only108.tfrecord'
    data_format = 'tfrecord'

    nasbench = api.NASBench(path_to_read_data,data_format=data_format)

    result_list = []

    with open('/home/strawberry/TUM/DLR/arch_specs.json', 'rb') as f:
        arch_specs = json.load(f)
    for arch_spec_name in evaluated_list_of_architectures:
        for i in range(len(arch_specs)):
            if arch_specs[i]['arch_code'] == arch_spec_name:
                matrix = arch_specs[i]['module_adjacency']
                ops = arch_specs[i]['module_operations']
                spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
                data = nasbench.query(spec)
                fixed_stats, computed_stats = nasbench.get_metrics_from_spec(spec)
                # print(computed_stats[108][0])
                print(computed_stats)
                exit()
                if val==True:
                    result_list.append(float(data['validation_accuracy']))
                else:
                    result_list.append(float(data['train_accuracy']))

    return result_list






# train_accuraciesLCZ = getAccuraciesLCZ(False)
import pickle
# with open('evaluated_list_of_architectures.pickle', 'wb') as handle:
#     pickle.dump(evaluated_list_of_architectures, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(train_accuraciesLCZ)

# val_accuraciesCifar = getCifarAccuracies(True)
# val_mean_CIFAR = mean_accuracy(val_accuraciesCifar)
# variance_CIFAR = variance(val_accuraciesCifar)
# print(val_accuraciesCifar)
# train_accuraciesCifar = getCifarAccuracies(False)


with open('/home/strawberry/TUM/DLR/tum-dlr-automl-for-eo/scripts/visualizations/results.pickle', 'rb') as handle:
    val_accuraciesCIFAR10 = pickle.load(handle)

def getMeanLCZ():
    val_accuraciesLCZ = getAccuraciesLCZ(True)
    val_mean_LCZ = mean_accuracy(val_accuraciesLCZ)
    print('mean LCZ:', val_mean_LCZ)
    return val_mean_LCZ

def getVarLCZ():
    val_accuraciesLCZ = getAccuraciesLCZ(True)
    variance_LCZ = variance(val_accuraciesLCZ)
    print('var LCZ:', variance_LCZ)
    return variance_LCZ

def getMeanCIFAR10():
    val_accuracies = np.array(val_accuraciesCIFAR10)[:,3]
    val_mean_CIFAR = mean_accuracy(val_accuracies)
    print('mean CIFAR:', val_mean_CIFAR)
    return val_mean_CIFAR

def getVarCIFIAR10():
    val_accuracies = np.array(val_accuraciesCIFAR10)[:, 3]
    variance_CIFAR = variance(val_accuracies)
    print('var CIFAR:', variance_CIFAR)
    return variance_CIFAR

def getPositivePersistanceAuCLCZ():
    val_accuraciesCifar = getAccuraciesLCZ(True,getLastOnly=False)
    pos_pers_auc = positive_persistence_auc(np.array(val_accuraciesCifar))
    print('positive persistance AuC LCZ:', pos_pers_auc)
    return pos_pers_auc


def getPositivePersistanceLCZ():
    val_accuraciesCifar = getAccuraciesLCZ(True,getLastOnly=False)
    pos_pers_auc = positive_persistence(np.array(val_accuraciesCifar))
    print('positive persistance LCZ:', pos_pers_auc)
    return pos_pers_auc

def getNegativePersistanceAuCLCZ():
    val_accuraciesCifar = getAccuraciesLCZ(True,getLastOnly=False)
    pos_pers_auc = negative_persistence_auc(np.array(val_accuraciesCifar))
    print('Negative persistance AuC LCZ:', pos_pers_auc)
    return pos_pers_auc


def getNegativePersistanceLCZ():
    val_accuraciesCifar = getAccuraciesLCZ(True,getLastOnly=False)
    pos_pers_auc = negative_persistence(np.array(val_accuraciesCifar))
    print('positive persistance AuC LCZ:', pos_pers_auc)
    return pos_pers_auc


def getPositivePersistanceAuCCIFAR():
    pos_pers_auc = positive_persistence_auc(np.array(val_accuraciesCIFAR10))
    print('positive persistance AuC CIFAR:', pos_pers_auc)
    return pos_pers_auc


def getPositivePersistanceCIFAR():
    pos_pers_auc = positive_persistence(np.array(val_accuraciesCIFAR10))
    print('positive persistance CIFAR:', pos_pers_auc)
    return pos_pers_auc

def getNegativePersistanceAuCCIFAR():
    pos_pers_auc = negative_persistence_auc(np.array(val_accuraciesCIFAR10))
    print('Negative persistance AuC CIFAR:', pos_pers_auc)
    return pos_pers_auc


def getNegativePersistanceCIFAR():
    pos_pers_auc = negative_persistence(np.array(val_accuraciesCIFAR10))
    print('positive persistance AuC CIFAR:', pos_pers_auc)
    return pos_pers_auc



# print('Mean validation accuracy So2Sat LCZ42: %.2f' % val_mean_LCZ)
# print('Variance validation accuracy So2Sat LCZ42: %.2f' % variance_LCZ)
# print('Mean validation accuracy CIFAR10: %.2f' % val_mean_CIFAR)
# print('Variance validation accuracy CIFAR10: %.2f' % variance_CIFAR)

# for i in range(len(val_accuraciesCifar)):
#     print('Architecture',list_of_architectures[i],'statistics:')
#     print('  - Train accuracy CIFAR10:%.2f' % train_accuraciesCifar[i])
#     print('  - Train accuracy So2Sat LCZ42:%.2f' % train_accuraciesLCZ[i])
#     print('  - Validation accuracy CIFAR10:%.2f' % val_accuraciesCifar[i])
#     print('  - Validation accuracy So2Sat LCZ42:%.2f' % val_accuraciesLCZ[i])