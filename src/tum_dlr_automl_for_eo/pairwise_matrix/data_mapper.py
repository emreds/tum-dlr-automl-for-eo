import numpy as np
import pandas as pd
from tum_dlr_automl_for_eo.pairwise_matrix.encode_matrix import ArchitectureEncoder


class NB101Mapper:
    """
    Maps the given nb101_dict to diferrent data structures.
    It is used to speed up the pairwise comparison.
    """
    
    def __init__(self, nb101_dict: dict):
        self.nb101_dict = nb101_dict
        self.encoder = ArchitectureEncoder()
        self.hash_arch_matrix = {}
        self.hash_to_id = {}
        self.id_to_hash = {}
        self.arch_keys = []
        self.hash_arch_array = np.array([])
    
    def map_data(self):
        """
        Maps the given nb101_dict to diferrent data structures.

        Returns:
            _type_: _description_
        """
        for binary_encode, arch_specs in self.nb101_dict.items():
            self.hash_arch_matrix[arch_specs["unique_hash"]] = self.encoder.encode_architecture(
                arch_specs["module_adjacency"], arch_specs["module_operations"]
            )
            
        for idx, hash_code in enumerate(self.hash_arch_matrix.keys()):
            self.hash_to_id[hash_code] = idx
            self.id_to_hash[idx] = hash_code

        self.arch_keys = list(self.hash_arch_matrix.keys())
        self.hash_arch_array = np.array([self.hash_arch_matrix[k].astype("uint8") for k in self.arch_keys])

        pass