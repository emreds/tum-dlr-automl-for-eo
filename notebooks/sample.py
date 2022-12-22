from pkgutil import iter_modules
import naslib

def list_submodules(module):
    for submodule in iter_modules(module.__path__):
        print(submodule.name)

list_submodules(naslib)

import logging
import sys


from naslib.search_spaces import (
    DartsSearchSpace,
    SimpleCellSearchSpace,
    NasBench101SearchSpace,
    HierarchicalSearchSpace,
)
from naslib.search_spaces.nasbench101 import graph

from naslib.utils import get_dataset_api, setup_logger, utils

import numpy as np


config = utils.get_config_from_args(config_type="nas")
dataset_api = get_dataset_api(config.search_space, config.dataset)

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9

def sample_random_architecture(dataset_api, arch_limit=10):
    """
    This will sample a random architecture and update the edges in the
    naslib object accordingly.
    From the NASBench repository:
    one-hot adjacency matrix
    draw [0,1] for each slot in the adjacency matrix
    """
    architectures = []
    Xs = []
    while len(architectures) < arch_limit:
        matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
        if dataset_api["nb101_data"].is_valid(spec):
            architectures.append(dataset_api["api"].ModelSpec(matrix=matrix, ops=ops))#{"matrix": matrix, "ops": ops})
            Xs.append(matrix.flatten())
            # break

    return architectures, Xs

sampled_architectures_train, X_train = sample_random_architecture(dataset_api,arch_limit=10)
sampled_architectures_test, X_test = sample_random_architecture(dataset_api,arch_limit=10)

from naslib.utils import nb101_api as api
path_to_read_data = '/home/strawberry/TUM/DLR/tum-dlr-automl-for-eo/notebooks/src/naslib/naslib/data/nasbench_only108.pkl'#'nasbench_only108.tfrecord'
data_format = 'tfrecord'

nasbench = api.NASBench(path_to_read_data)
data = nasbench.query(sampled_architectures_train[0])
print(data)

Y_train = []
Y_test = []
for arch in sampled_architectures_train:
    validation_accuracy = nasbench.query(arch)['validation_accuracy']
    Y_train.append(validation_accuracy)

for arch in sampled_architectures_test:
    validation_accuracy = nasbench.query(arch)['validation_accuracy']
    Y_test.append(validation_accuracy)

import xgboost as xgb
from sklearn import metrics

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


model = xgb.XGBRegressor()
model.fit(X_train, Y_train)

predicted_y = model.predict(X_test)
print(metrics.mean_squared_log_error(Y_test, predicted_y))
