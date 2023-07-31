import numpy as np
from tum_dlr_automl_for_eo.pairwise_matrix.data_mapper import NB101Mapper
from tum_dlr_automl_for_eo.utils import file
from tum_dlr_automl_for_eo.pairwise_matrix.dist_calculator import calculate_sample_pairwise_dist
from tum_dlr_automl_for_eo.pairwise_matrix.collect_existing_archs import CollectTrainedArchs
import random


class PathSampler:
    
    def __init__(self, starting_ids, steps, hash_arch_array, no_start_samples=set(), trained_ids=set()):
        self.steps = steps
        self.starting_ids = set(starting_ids) - no_start_samples
        self.trained_ids = trained_ids
        self.hash_arch_array = hash_arch_array
    
    def sampler(self):
        
        paths = {s_id: [] for s_id in self.starting_ids}
        picked_ids = set()
        picked_ids |= self.starting_ids

        for sample_id in sorted(self.starting_ids): 
            #sample_hash = id_to_hash[sample_id]
            curr_id = sample_id
            for i in range(self.steps):
                curr_row = self.hash_arch_array[curr_id,:,:]
                # Distances has shape [1,423624]
                distances = calculate_sample_pairwise_dist(curr_row, self.hash_arch_array)
                # np.where returns tuple here, what we want is `[0]`
                neighbor_ids = np.where(distances[0,:] == 1)[0]
                # If one of the neighbors is in the trained_archs, we pick that arch.
                trained_new = set(neighbor_ids).intersection(self.trained_ids).difference(picked_ids)
                if trained_new:
                    curr_id = trained_new.pop()
                    paths[sample_id].append(curr_id)
                    picked_ids.add(curr_id)
                else:
                    for nei_id in neighbor_ids:
                        if nei_id not in picked_ids:
                                curr_id = nei_id
                                paths[sample_id].append(curr_id)
                                picked_ids.add(curr_id)
                                break
        print("Number of picked ids are: ", len(picked_ids))
        return paths
    
def path_refiner(paths, required_len=30, trained_ids=set()):
    '''
    Filters the number of paths to get the paths with most trained architecture.
    '''
    refined_paths = []
    path_trained_cnt = {src:0 for src in paths}

    for src, vals in paths.items(): 
        path_trained_cnt[src] = len(set(vals).intersection(trained_ids))
    
    picked_ids = set()
    # Checks if the path contains at least one already trained architecture.
    for arch_id, path_len in path_trained_cnt.items():
        if path_len > 0:
            full_path = [arch_id] + paths[arch_id]
            refined_paths.append(full_path)
            picked_ids.add(arch_id)
            
    if len(refined_paths) < required_len:
        unpicked_paths = set(paths.keys()).difference(picked_ids)
        random_picked = random.choices(list(unpicked_paths), k=(required_len - len(refined_paths)))
    
    for arch_id in random_picked:
        full_path = [arch_id] + paths[arch_id]
        refined_paths.append(full_path)     
    
    return refined_paths


def arch_logger(sequences, id_to_hash, nb101_dict):
    """
    Creates a dictionary for the sampled paths coupling them with unique hash codes 
    and module adjacency and operations.

    Args:
        sequences (_type_): _description_
        id_to_hash (_type_): _description_
        nb101_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    sequences_hash = []
    
    for sequence in sequences:
        sequences_hash.append([(id_to_hash[idx], idx) for idx in sequence])
        
    logs = []
    for sequence in sequences_hash:
        sub_logs = []
        for step, (hash_code, idx) in enumerate(sequence):
            for binary_encode, arch_specs in nb101_dict.items():
                if arch_specs["unique_hash"] == hash_code:
                    sub_logs.append({"id": idx, "unique_hash": hash_code, "step": step, "module_adjacency": arch_specs["module_adjacency"], "module_operations": arch_specs["module_operations"]})
                    break
        logs.append(sub_logs)
        
    return logs
    