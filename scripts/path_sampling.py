import json
import os
import random

from tum_dlr_automl_for_eo.pairwise_matrix.data_mapper import NB101Mapper
from tum_dlr_automl_for_eo.pairwise_matrix.collect_existing_archs import CollectTrainedArchs
from tum_dlr_automl_for_eo.pairwise_matrix.path_sampler import PathSampler, path_refiner, arch_logger
from tum_dlr_automl_for_eo.utils import file

import torch

from naslib.predictors.utils.models import nasbench1 as nas101_arch
from naslib.predictors.utils.models import nasbench1_spec
from naslib.utils import utils as naslib_utils

def generate_paths():
        if not os.path.exists("./path_logs.json"):
            nb101_dict = file.load_pickle("/p/project/hai_nasb_eo/emre/arch_matrix/nb101_dict")
            nb101_mapper = NB101Mapper(nb101_dict)
            nb101_mapper.map_data()
            id_to_hash = nb101_mapper.id_to_hash
            
            # We have made some trials and those no pick samples goes a shorter way then the required amount of steps.
            no_start_samples = set([23596, 150491, 191805, 221281, 309560, 342022, 350831])
        
            if not os.path.exists("./picked_paths.json"):
                hash_arch_array = nb101_mapper.hash_arch_array
                hash_to_id = nb101_mapper.hash_to_id
                # The part below is not necessary for everyone, we had some already trained architectures and we are going to sample 
                # the starting points from those already trained architectures.
                all_samples_path = "../../arch_matrix/arch_specs.json"
                sequences_path = "../../arch_matrix/sequences.json"
                test_results_path = "../../arch_matrix/test_results_all.json"
                
                collect_archs = CollectTrainedArchs(all_samples_path=all_samples_path, sequences_path=sequences_path, test_results_path=test_results_path)
                trained_ids = collect_archs.get_trained_ids(hash_to_id)
                starting_ids = collect_archs.get_starting_ids(hash_to_id)
                sampler = PathSampler(starting_ids=starting_ids, steps=path_length, hash_arch_array=hash_arch_array, no_start_samples=no_start_samples, trained_ids=trained_ids)
                paths = sampler.sampler()
                #print(f"Those are the sampled paths: {paths}")
                refined_paths = path_refiner(paths, source_count, trained_ids)
                with open("./picked_paths.json", 'w') as f:
                    f.write(json.dumps(refined_paths, cls=file.NumpyArrayEncoder))
            else:
                refined_paths = file.load_json("./picked_paths.json")
            
            logs = arch_logger(refined_paths, id_to_hash, nb101_dict)
            
            with open("./path_logs.json", 'w') as f:
                f.write(json.dumps(logs, cls=file.NumpyArrayEncoder))
        else:
            logs = file.load_json("./path_logs.json")
                
        return logs

def create_torch_files(logs, export_path):
        nas_config = naslib_utils.get_config_from_args(config_type="nas")
        out_channel = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120, "So2Sat": 17}
        input_channel = {"cifar10": 3, "Sentinel-1": 8, "Sentinel-2": 10}
        
        
        for sequence in logs: 
            for arch in sequence:
                spec = nasbench1_spec._ToModelSpec(arch["module_adjacency"], arch["module_operations"])
                torch_arch = nas101_arch.Network(
                    spec,
                    stem_out=128,
                    stem_in=input_channel["Sentinel-2"],
                    num_stacks=3,
                    num_mods=3,
                    num_classes=out_channel["So2Sat"],
                )
                arch_file = "arch_" +  str(arch["id"]) + ".pt"
                arch_path = export_path + "/" + arch_file
                torch.save(torch_arch, arch_path, )



if __name__ == "__main__":
    general_config = file.get_general_config()
    random.seed(general_config["random_seed"])
    source_count = 30
    path_length = 14
    logs = generate_paths()
    export_path = "/p/project/hai_nasb_eo/sampled_paths/untrained_archs"
    # --result, export_path
    
