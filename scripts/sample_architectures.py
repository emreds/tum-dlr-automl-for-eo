
from naslib.utils import get_dataset_api, utils

import numpy as np

from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.utils import utils

from pathlib import Path

import torch
import os

import json


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
    while len(architectures) < arch_limit:
        matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
        if dataset_api["nb101_data"].is_valid(spec):
            architectures.append({"matrix": matrix, "ops": ops})
            # break

    return architectures


if __name__ == "__main__":
    config = utils.get_config_from_args(config_type="nas")
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    sampled_architectures = sample_random_architecture(dataset_api=dataset_api)
    os.mkdir("./architectures")
    
    out_channel = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120, "So2Sat":17}
    input_channel = {"cifar10": 3, "Sentinel-1":8, "Sentinel-2": 10}
    model_dict = {}
    
    for num, arch in enumerate(sampled_architectures):
        spec = nasbench1_spec._ToModelSpec(arch["matrix"], arch["ops"])

        torch_arch = nas101_arch.Network(
            spec,
            stem_out=128,
            stem_in=input_channel["Sentinel-2"],
            num_stacks=3,
            num_mods=3,
            num_classes=out_channel["So2Sat"],
        )
        
        
        model_dict[str(num)] = {"matrix": arch["matrix"], "ops": arch["ops"]}
        
        
        arch_path = "./architectures/arch_" + str(num)
        torch.save(torch_arch, arch_path, )
        
        
    with open("model_dict.json", 'w') as f:
        f.write(json.dumps(model_dict, cls=utils.NumpyArrayEncoder))
