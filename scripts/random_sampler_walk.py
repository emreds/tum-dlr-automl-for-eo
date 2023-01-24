import numpy as np
import pickle

from naslib.utils import get_dataset_api, utils
from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.utils import utils
import torch
import os
import json

path_to_read_nb101_dict = "../../nb101_dict"
number_of_arc_to_sample = 100  # num of archs to sample
all_archs = set([])
export_path = "./sampled_archs"
ENCODING_LEN = 289  # fixed encoding length
NUM_OF_STEPS = 7  # number of steps to walk

def get_nb101():
    with open(path_to_read_nb101_dict, 'rb') as f:
        nb101_dict = pickle.load(f)
    
    return nb101_dict

def random_walk(architecture, max_steps):
    step = 0
    curr_architecture = architecture
    while step < max_steps:
        random_nei_index = np.random.choice(range(0, ENCODING_LEN), size=1)[0]
        if curr_architecture[random_nei_index] == '0':
            found_architecture = curr_architecture[:random_nei_index] + '1' + curr_architecture[random_nei_index + 1:]
        else:
            found_architecture = curr_architecture[:random_nei_index] + '0' + curr_architecture[random_nei_index + 1:]
        
        if found_architecture in all_archs: 
            print("its in all archs")
        #print(f"This is nb get's results {nb101_dict.get(found_architecture)}")
        if (nb101_dict.get(found_architecture) is not None) and (found_architecture not in all_archs):
            all_archs.add(found_architecture)
            curr_architecture = found_architecture
            step += 1
    pass


def sample_random_archs(number_of_samples, key_dict):
    keys = list(key_dict.keys())
    sampled_keys = np.random.choice(keys, size=number_of_samples, replace=False)
    return sampled_keys

## There should be a global dict which keeps track of all the samples. 
## We need to add every architecture's key to that dict or set to not train same architecture twice. 
## Even if it's part of the random walk, the step should be in different direction.

if __name__ == "__main__":
    nb101_dict = get_nb101()

    random_sampled_archs = sample_random_archs(number_of_samples=number_of_arc_to_sample, key_dict=nb101_dict)
    config = utils.get_config_from_args(config_type="nas")
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    out_channel = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120, "So2Sat": 17}
    input_channel = {"cifar10": 3, "Sentinel-1": 8, "Sentinel-2": 10}
    
    
    os.mkdir(export_path)
    
    for sampled_arch in random_sampled_archs:
        random_walk(sampled_arch, NUM_OF_STEPS)
    
    arch_specs = {}
    for idx, arch in enumerate(all_archs):
        arch_spec = nb101_dict[arch]
        arch_spec["arch_code"] = "arch_" + str(idx)
        spec = nasbench1_spec._ToModelSpec(arch_spec['module_adjacency'], arch_spec['module_operations'])
        torch_arch = nas101_arch.Network(
            spec,
            stem_out=128,
            stem_in=input_channel["Sentinel-2"],
            num_stacks=3,
            num_mods=3,
            num_classes=out_channel["So2Sat"],
        )
         
        arch_path = "./sampled_archs/" +  arch_spec["arch_code"]
        torch.save(torch_arch, arch_path, )
        
    arch_specs[arch] = arch_spec
        
    with open("./sampled_archs/arch_specs.json", 'w') as f:
        f.write(json.dumps(arch_specs, cls=utils.NumpyArrayEncoder))
