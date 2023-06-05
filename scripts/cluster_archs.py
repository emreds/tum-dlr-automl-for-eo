import json
from collections import defaultdict
from typing import List, Tuple

import numpy as np

RANDOM_SEED = 42
START_SAMPLE_COUNT = 100
LOCAL_SEARCH_STEPS = 7

np.random.seed(RANDOM_SEED)

#arch_specs_path = "../../../training/sampled_archs/arch_specs.json"
arch_specs_path = "/p/project/hai_nasb_eo/training/sampled_archs/arch_specs.json"

class Arch:
    """
    Architecture class to hold architecture properties.
    """
    def __init__(self, arch_idx, binary): 
        self.index = arch_idx
        self.binary_encode = binary

class Cluster:
    """
    Cluster class to group architectures.
    """
    def __init__(self, arch:Arch): 
        self.min_arch_idx = arch.index 
        self.base_arch = arch.binary_encode
        self.neighbours = [arch]
        self.length = 1
        
    def add_arch(self, arch:Arch):
        """
        Adds the given architecture to the cluster. 
        Changes the min_index of the cluster if the architecture has lower index.

        Args:
            arch (Arch)
        """

        if arch.index < self.min_arch_idx: 
            self.min_arch_idx = arch.index
        
        self.neighbours.append(arch)
        self.length += 1

def is_in_cluster(cluster_bin: str, arch_bin: str) -> bool:
    """
    Compares if the binary representation of the architecture is near enough the cluster.

    Args:
        cluster_bin (str)
        arch_bin (str)

    Returns:
        bool
    """
    # Calculate the bitwise difference using XOR
    diff = int(cluster_bin, 2) ^ int(arch_bin, 2)

    # Count the number of set bits (1s)
    count = bin(diff).count("1")

    return count <= LOCAL_SEARCH_STEPS

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
    

def get_arch_specs() -> List[dict]:
    """
    Reads the arch specs and puts it into a list.

    Returns:
        List[dict]
    """

    with open(arch_specs_path) as json_file:
        raw_archs = json.load(json_file)
        
    return raw_archs

def get_clusters_archs(raw_archs, cluster_idx) -> Tuple[List[Cluster], List[Arch]]:
    """
    Picks the given indexes as clusters from raw_archs and the rest becomes architectures to group.

    Args:
        raw_archs (List): Architecture dictionary.
        cluster_idx (List): Cluster index list.

    Returns:
        clusters, archs: List[Cluster], List[Arch]
    """
    clusters = []
    archs = []
    for raw_arch in raw_archs: 
        arch_idx = int(raw_arch["arch_code"].split("_")[-1])
        arch = Arch(arch_idx, raw_arch["binary_encoded"])
        if arch_idx not in cluster_idx:
            archs.append(arch)
        else:
            clusters.append(Cluster(arch))
    
    return clusters, archs

def match(archs, clusters, cluster_idx) -> List[Arch]:
    """
    Matches the given architectures with the respective clusters.

    Args:
        archs (List[Arch])
        clusters (List[Cluster])
        cluster_idx (Set[String])

    Returns:
        List[Arch]: List of architectures with no match.
    """
    no_match = []
    for arch in archs: 
        if arch.index not in cluster_idx:
            match = False
            for cluster in clusters:
                if cluster.length < LOCAL_SEARCH_STEPS+1 and is_in_cluster(cluster_bin=cluster.base_arch, arch_bin=arch.binary_encode):
                    cluster.add_arch(arch)
                    match = True
                    break
            if not match: 
                no_match.append(arch)
                
    return no_match

if __name__ == "__main__":
    raw_archs = get_arch_specs()
    #print(raw_archs)
    added_archs = set([])
    arch_diffs = {}
    for arch in raw_archs:
        arch_code = arch["arch_code"]
        arch_diffs[arch_code] = [0, 0, defaultdict(list)]
        for other_arch in raw_archs:
            other_arch_code = other_arch["arch_code"] 
            if arch_code != other_arch_code:
                dist = hamming_distance(arch["binary_encoded"], other_arch["binary_encoded"])
                # We check if the hamming distance is lower than 7. 
                # If the new architecture was not already added as a neighbour to an another architecture. 
                # We check if there is no neighbour for that distance. So for every distance there is 1 neighbour.
                if dist <= 7 and other_arch_code not in added_archs and not arch_diffs[arch_code][2][dist]:
                    new_dist = max(arch_diffs[arch_code][0], dist)
                    cnt = arch_diffs[arch_code][1] + 1
                    arch_diffs[arch_code][0] = new_dist
                    arch_diffs[arch_code][1] = cnt
                    arch_diffs[arch_code][2][dist].append(other_arch_code)
                    added_archs.add(other_arch_code)
    
    

    max_diff = 0 
    for val in arch_diffs.values(): 
        for d_val in val[2].keys(): 
            if d_val > max_diff:
                max_diff = d_val
    #max_diff = max([max(list(val[2].keys()).append(0)) for val in arch_diffs.values()], 0)
    #print(arch_diffs)
    
    roots = []
    for key, value in arch_diffs.items():
        # For the architectures that I cannot find more than 4 neighbours. I subtract the number.
        if len(value[2]) >= max_diff - 3: 
            roots.append(key)
            
    #print(roots)
    print(f"This is the max diff: {max_diff}")
    print(f"This is len roots: {len(roots)}")
    print(arch_diffs[roots[0]])
    #print(len(arch_diffs.keys()))        
    #print(roots)
    #print(arch_diffs['arch_91'])
    #print(arch_diffs['arch_7'])
        
    #cluster_idx = set(np.random.randint(0, len(raw_archs), START_SAMPLE_COUNT))
    #clusters, archs = get_clusters_archs(raw_archs=raw_archs, cluster_idx=cluster_idx)
    #no_match = match(archs, clusters=clusters)
    #for cluster in clusters: 
    #    print(f"Length of this cluster is {cluster.length}")
    # print(f"There are {len(no_match)} architectures which are not in any cluster")
    # print(f"There are {len(no_match)} architectures which are not in any cluster")
