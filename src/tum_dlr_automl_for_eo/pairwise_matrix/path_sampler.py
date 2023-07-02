import numpy as np
from data_mapper import NB101Mapper
from tum_dlr_automl_for_eo.utils import file
from dist_calculator import calculate_sample_pairwise_dist
from collect_existing_archs import CollectTrainedArchs
import random
import json



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
                curr_row = hash_arch_array[curr_id,:,:]
                # Distances has shape [1,423624]
                distances = calculate_sample_pairwise_dist(curr_row, hash_arch_array)
                # np.where returns tuple here, what we want is `[0]`
                neighbor_ids = np.where(distances[0,:] == 1)[0]
                # If one of the neighbors is in the trained_archs, we pick that arch.
                trained_new = set(neighbor_ids).intersection(trained_ids).difference(picked_ids)
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

        return paths
    
def path_refiner(paths, required_len=30, trained_ids=set()):
    '''
    Filters the number of paths to get the paths with most trained architecture.
    '''
    refined_paths = []
    path_trained_cnt = {src:0 for src in paths}

    for src, vals in paths.items(): 
        path_trained_cnt[src] = len(set(vals).intersection(trained_ids))
    
    # Checks if the path contains at least one already trained architecture.
    for arch_id, path_len in path_trained_cnt.items():
        if path_len > 0:
            refined_paths.append(paths[arch_id])
            
    if len(refined_paths) < required_len:
        unpicked_paths = set(paths.keys()).difference(set(path_trained_cnt.keys()))
        random_picked = random.choice(unpicked_paths, required_len - len(refined_paths))
    
    refined_paths += random_picked
    
    return refined_paths
    
if __name__ == "__main__":
    config = file.get_general_config()
    random.seed(config["random_seed"])
    nb101_dict = file.load_pickle("/p/project/hai_nasb_eo/emre/arch_matrix/nb101_dict")
    nb101_mapper = NB101Mapper(nb101_dict)
    nb101_mapper.map_data()
    hash_arch_array = nb101_mapper.hash_arch_array
    hash_to_id = nb101_mapper.hash_to_id
    # The part below is not necessary for everyone, we had some already trained architectures and we are going to sample 
    # the starting points from those already trained architectures.
    all_samples_path = "../../../../arch_matrix/arch_specs.json"
    sequences_path = "../../../../arch_matrix/sequences.json"
    test_results_path = "../../../../arch_matrix/test_results_all.json"
    # We have made some trials and those no pick samples goes a shorter way.
    no_start_samples = set([23596, 150491, 191805, 221281, 309560, 342022, 350831])
    collect_archs = CollectTrainedArchs(all_samples_path, sequences_path, test_results_path)
    trained_ids = collect_archs.get_trained_ids(hash_to_id)
    starting_ids_func = collect_archs.get_starting_ids(hash_to_id)
    sampler = PathSampler(starting_ids_func, 14, hash_arch_array, no_start_samples, trained_ids)
    paths = sampler.sampler()
    #print(f"Those are the sampled paths: {paths}")
    refined_paths = path_refiner(paths, 30, trained_ids)
    with open("./picked_paths.json", 'w') as f: 
        json.dump(refined_paths, f)