import numpy as np
from data_mapper import NB101Mapper
from encode_matrix import ArchitectureEncoder
from tum_dlr_automl_for_eo.utils import file
from dist_calculator import calculate_sample_pairwise_dist



class PathSampler:
    
    def __init__(self, starting_ids, steps, hash_arch_array, no_pick_samples=[], trained_ids=set()):
        self.steps = steps
        self.starting_ids = starting_ids
        self.no_pick_samples = no_pick_samples
        self.trained_ids = trained_ids
        self.hash_arch_array = hash_arch_array
    
    def sampler():
        
        paths = {s_id: [] for s_id in starting_ids}
        picked_ids = set()
        picked_ids |= set(self.starting_ids)
        picked_ids ^= set(self.no_pick_samples)

        for sample_id in self.starting_ids: 
            sample_hash = id_to_hash[sample_id]
            curr_id = sample_id
            for i in range(self.steps):
                curr_row = hash_arch_array[curr_id,:,:]
                # Distances has shape [1,423624]
                distances = calculate_pairwise_simple_dist(curr_row, hash_arch_array)
                # np.where returns tuple here, what we want is `[0]`
                neighbor_ids = np.where(distances[0,:] == 1)[0]
                # If one of the neighbors is in the trained_archs, we pick that arch.
                if trained_ids:
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
    
def path_refiner(paths, reqired_len):
    '''
    Filters the number of paths to get the paths with most trained architecture.
    '''
    zeros = 0
    for arch_id, val in path_trained_cnt.items():
        if val > 0  and arch_id not in no_pick_samples: 
            zeros += 1
    
    
    
if __name__ == "__main__":

    nb101_dict = file.load_pickle("/p/project/hai_nasb_eo/emre/arch_matrix/nb101_dict")
    nb101_mapper = NB101Mapper(nb101_dict)
    nb101_mapper.map_data()
    # The part below is not necessary for everyone, we had some already trained architectures and we are going to sample 
    # the starting points from those already trained architectures.
    all_existing_archs = get_json_file("./arch_specs.json")
    sequences = get_json_file("./sequences.json")
    trained_archs = get_json_file("./test_results_all.json")
    # We have made some trials and those no pick samples goes a shorter way.
    no_pick_samples = set([23596, 150491, 191805, 221281, 309560, 342022, 350831])