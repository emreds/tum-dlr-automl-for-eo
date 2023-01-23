import numpy as np
import pickle

from naslib.utils import get_dataset_api, utils
from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.utils import utils
import torch
import os
# give required paths to read and save files


path_to_read_nb101_dict = "/Users/safayilmaz/Desktop/DI LAB/NASLib/naslib/data/nb101_dict"
number_of_arc_to_sample = 3  # num of archs to sample
ENCODING_LEN = 289  # fixed encoding length
NUM_OF_STEPS = 7  # number of steps to walk


def random_walk(architecture):
    found_keys = list()
    curr_architecture = architecture
    while len(found_keys) < NUM_OF_STEPS:
        random_nei_index = np.random.choice(range(0, ENCODING_LEN), size=1)[0]
        if curr_architecture[random_nei_index] == '0':
            found_architecture = curr_architecture[:random_nei_index] + '1' + curr_architecture[random_nei_index + 1:]
        else:
            found_architecture = curr_architecture[:random_nei_index] + '0' + curr_architecture[random_nei_index + 1:]
        if nb101_dict.get(found_architecture) is not None and found_architecture not in found_keys:
            found_keys.append(found_architecture)
            curr_architecture = found_architecture
    return found_keys


def sample_random_keys(number_of_samples):
    keys = list(nb101_dict.keys())
    sampled_keys = np.random.choice(keys, size=number_of_samples)
    return sampled_keys


if __name__ == "__main__":
    with open(path_to_read_nb101_dict, 'rb') as f:
        global nb101_dict
        nb101_dict = pickle.load(f)

    random_sampled_keys = sample_random_keys(number_of_samples=number_of_arc_to_sample)
    config = utils.get_config_from_args(config_type="nas")
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    out_channel = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120, "So2Sat": 17}
    input_channel = {"cifar10": 3, "Sentinel-1": 8, "Sentinel-2": 10}
    os.mkdir("./sampled_points")
    for sampled_key_index, key in enumerate(random_sampled_keys):
        os.mkdir("./sampling_" + str(sampled_key_index))
        random_keys = list()
        random_keys.append(key)
        # print('random walk started')
        random_walk_res = random_walk(key)
        # print('random walk completed')

        # bring staring points and random walks together
        random_keys += random_walk_res
        # save random walk keys
        sampling_path  = "./sampled_points" + "/random_walk_" + str(sampled_key_index)
        filehandler = open(sampling_path, 'wb')
        pickle.dump(random_keys, filehandler)

        for random_key_index, random_key in enumerate(random_keys):
            arch = nb101_dict[random_key]

            spec = nasbench1_spec._ToModelSpec(arch['module_adjacency'], arch['module_operations'])
            torch_arch = nas101_arch.Network(
                spec,
                stem_out=128,
                stem_in=input_channel["Sentinel-2"],
                num_stacks=3,
                num_mods=3,
                num_classes=out_channel["So2Sat"],
            )
            arch_path = "./sampling_" + str(sampled_key_index) + "/arch_" + str(random_key_index)
            torch.save(torch_arch, arch_path, )

