import numpy as np
import pickle

# give required paths to read and save files

path_to_save_random_walks = "/Users/safayilmaz/Desktop/DI LAB/NASLib/naslib/data/"
path_to_read_nb101_dict = "/Users/safayilmaz/Desktop/DI LAB/NASLib/naslib/data/nb101_dict"
number_of_arc_to_sample = 10  # num of archs to sample
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

    for index, key in enumerate(random_sampled_keys):
        random_keys = list()
        random_keys.append(key)
        # print('random walk started')
        random_walk_res = random_walk(key)
        # print('random walk completed')
        random_keys.append(random_walk_res)
        path = path_to_save_random_walks + "random_walk_" + str(index)
        filehandler = open(path, 'wb')
        pickle.dump(random_keys, filehandler)
