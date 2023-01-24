import numpy as np
import pickle

from naslib.utils import get_dataset_api, utils
from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.utils import utils
import torch
import os
import json
import random

path_to_read_nb101_dict = "../../nb101_dict"
number_of_arc_to_sample = 100  # num of archs to sample
all_archs = set([])
export_path = "./sampled_archs"
ENCODING_LEN = 289  # fixed encoding length
NUM_OF_STEPS = 7 # number of steps to walk

def get_nb101():
    with open(path_to_read_nb101_dict, 'rb') as f:
        nb101_dict = pickle.load(f)
    
    return nb101_dict

def random_walk(architecture, max_steps, hamming_distance=2):
    """

    Args:
        architecture (_type_): _description_
        max_steps (_type_): _description_
        hamming_distance (_type_): _description_
    """    
    step = 0
    curr_architecture = architecture
    random_flag = 0
    stuck_count = 0
    arch_len = len(all_archs)
    while step < max_steps and stuck_count <= ENCODING_LEN:
        rand_nums = np.random.choice(range(0, ENCODING_LEN), size=hamming_distance, replace=False)
        print(f"This is rand_nums: {rand_nums}")
        found_architecture = curr_architecture
        for random_nei_index in rand_nums:
            if found_architecture[random_nei_index] == '0':
                found_architecture = found_architecture[:random_nei_index] + '1' + found_architecture[random_nei_index + 1:]
            else:
                found_architecture = found_architecture[:random_nei_index] + '0' + found_architecture[random_nei_index + 1:]
            
            if (nb101_dict.get(found_architecture) is not None) and (found_architecture not in all_archs):
                all_archs.add(found_architecture)
                curr_architecture = found_architecture
                step += 1
                arch_len += 1
                break
                
        
        if len(all_archs) == arch_len: 
            print("its not adding an architecture")
            hashed = abs(hash(found_architecture)) % (10 ** 8)
            print(f"This is the arc: {hashed}")
            print(f"Length of all archs: {len(all_archs)}")
            print(f"This is the random_index: {rand_nums}")
            print(f"Passing the arc, len of arcs: {len(all_archs)}")
            stuck_count += 1
    pass


def sample_random_archs(number_of_samples, key_dict):
    keys = list(key_dict.keys())
    sampled_keys = np.random.choice(keys, size=number_of_samples, replace=False)
    return sampled_keys

if __name__ == "__main__":
    nb101_dict = get_nb101()

    random_sampled_archs = sample_random_archs(number_of_samples=number_of_arc_to_sample, key_dict=nb101_dict)
    config = utils.get_config_from_args(config_type="nas")
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    out_channel = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120, "So2Sat": 17}
    input_channel = {"cifar10": 3, "Sentinel-1": 8, "Sentinel-2": 10}
    
    for sampled_arch in random_sampled_archs:
        random_walk(sampled_arch, NUM_OF_STEPS)
        
    os.mkdir(export_path)
    arch_specs = []
    print(f"Out of the random walk loop, len of archs: {len(all_archs)}")
    for idx, arch in enumerate(all_archs):
        arch_spec = {}
        arch_spec = nb101_dict[arch]
        arch_spec["binary_encoded"] = arch
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
        
        arch_specs.append(arch_spec)
        
    with open("./sampled_archs/arch_specs.json", 'w') as f:
        f.write(json.dumps(arch_specs, cls=utils.NumpyArrayEncoder))
