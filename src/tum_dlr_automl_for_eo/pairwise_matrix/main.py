import numpy as np
from data_mapper import NB101Mapper
from encode_matrix import ArchitectureEncoder
from tum_dlr_automl_for_eo.utils import file
from dist_calculator import calculate_pairwise_dist

if __name__ == "__main__":
    nb101_dict = file.load_pickle("/p/project/hai_nasb_eo/emre/arch_matrix/nb101_dict")
    nb101_mapper = NB101Mapper(nb101_dict)
    nb101_mapper.map_data()
    dist_matrix = np.zeros((1, len(nb101_mapper.hash_arch_array)), dtype="uint8")
    #TODO: Don't forget to add last 624 architectures in hash_arch_array
    splits = np.split(nb101_mapper.hash_arch_array[:423000, :, :], 423, axis=0)
    last_624 = nb101_mapper.hash_arch_array[423000::]
                           
    for split in splits[:5]:
        dist_chunk = calculate_pairwise_dist(split, nb101_mapper.hash_arch_array)
        dist_matrix = np.concatenate((dist_matrix, dist_chunk), axis=0)
    
    dist_chunk = calculate_pairwise_dist(last_624, nb101_mapper.hash_arch_array)
    dist_matrix = np.concatenate((dist_matrix, dist_chunk), axis=0)
    
    dist_matrix = dist_matrix[1:]
    
    np.save("/p/project/hai_nasb_eo/emre/arch_matrix/dist_matrix.npy", dist_matrix)
    
    
    