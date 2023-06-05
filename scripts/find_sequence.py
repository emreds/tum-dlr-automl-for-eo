import json
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np

arch_specs_path = "/p/project/hai_nasb_eo/training/sampled_archs/arch_specs.json"

def hamming_distance(bin_enc_1: str, bin_enc_2: str) -> int:
    """
    Returns the hamming distance between 2 binary encoded strings.

    Args:
        bin_enc_1 (str)
        bin_enc_2 (str)

    Returns:
        int: Hamming distance.
    """
    # Calculate the bitwise difference using XOR
    diff = int(bin_enc_1, 2) ^ int(bin_enc_2, 2)

    # Count the number of set bits (1s)
    distance = bin(diff).count("1")
    # slowride: sum(c1 != c2 for c1, c2 in zip(bin_enc_1, bin_enc_2))
    return distance
    

def get_arch_specs(path) -> List[dict]:
    """
    Reads the arch specs and puts it into a list.

    Returns:
        List[dict]
    """

    with open(path) as json_file:
        raw_archs = json.load(json_file)
        
    return raw_archs

def find_dist_neighbours(raw_archs):
        sequences = {}
        
        # I want to see what would be the maximum path length for an architecture. 
        for arch in raw_archs: 
            code = arch['arch_code']
            sequences[code] = {}
            curr_dist = 1
            for other_arch in raw_archs:
                other_code = other_arch["arch_code"] 
                if code != other_code:
                    dist = hamming_distance(arch["binary_encoded"], other_arch["binary_encoded"])
                    if dist == curr_dist:
                        tag = str(curr_dist) + "_step"
                        sequences[code][tag] = {other_code: other_arch}
                        curr_dist += 1
                        
        return sequences
    
def search_arch(arch, all_archs, excluded, dist=1):
    res = None
    for candidate in all_archs:
        candidate_code = candidate["arch_code"] 
        arch_dist = hamming_distance(arch["binary_encoded"], candidate["binary_encoded"])
        if arch_dist == dist and (candidate_code not in excluded):
            return candidate
    
    return res

def find_path(raw_archs):
    
    # take all archs.
    # start with arch 0
    # create a set of sampled archs add curr arch
    # search step 1 arch in all archs 
    # when you find a step x neigh which is not in sampled archs
        # # add it as step x neigh to dict 
        # # increase step x += 1
        # # then, make the found arch = curr_arch
        # # add new arch to sampled archs.
    # search again with curr_arch 
    # after the search if you cannot find the arch, then break the search. 
    # leave that sequence as it is.
    sequences = {}
    
    for arch in raw_archs: 
        code = arch['arch_code']
        sequences[code] = {}
        curr_step = 1
        curr_arch = arch
        sampled = set([code])
        
        found = True
        while found: 
            res = search_arch(curr_arch, raw_archs, sampled)
            if res:
                tag = str(curr_step) + "_step"
                sequences[code][tag] = res             
                curr_step += 1
                curr_arch = res
                sampled.add(res["arch_code"])
            else: 
                found = False
    
    return sequences
    
    
def write_dict_to_json(dict, path): 
    with open(path, 'w') as file: 
        json.dump(dict, file)
    
    
if __name__ == "__main__": 
    arch_specs_path = "/p/project/hai_nasb_eo/training/sampled_archs/arch_specs.json"
    raw_archs = get_arch_specs(arch_specs_path)
    #sequences = find_dist_neighbours(raw_archs)
    
    
    sequences = find_path(raw_archs)
    
    seq_lens = [len(sequences[arch]) for arch in sequences]
    write_dict_to_json(sequences, './sequences.json')
    print(sorted(Counter(seq_lens).items()))
    print(sum(Counter(seq_lens).values()))
    
        
    #print(sequences['arch_0'])

    