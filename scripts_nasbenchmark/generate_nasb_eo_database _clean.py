#!/usr/bin/env
import json
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    arch_str_prefix = 'arch_'
    path = '/p/project/hai_nasb_eo/sampled_paths/all_trained_archs/'

    ### collect all architectures ids in directory
    # list of all content in a directory, filtered so only directories are returned
    dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
    # deal with duplicates
    dir_list_updated = [dir_local for dir_local in dir_list if len(set(dir_local.split('_'))) == len(dir_local.split("_"))] 
    
    file_test_results = '../nasbench_database/test_results_sampled_all_paths.json'
    file_specs_archs = '../nasbench_database/path_logs.json'
    
    dict_database_to_pickle_macro = dict()    
    dict_database_to_pickle_micro = dict()    
    dict_database_to_pickle_latency = dict()   
    dict_database_to_pickle_MAC = dict()   
    
    
    pickle_file_for_database_micro =  '../nasbench_database/database_pickle_micro.pickle'
    pickle_file_for_database_macro =  '../nasbench_database/database_pickle_macro.pickle'
    pickle_file_for_database_latency =  '../nasbench_database/database_pickle_latency.pickle'
    pickle_file_for_database_MAC =  '../nasbench_database/database_pickle_MACs.pickle'
    
    cpt_empty_evals = 0
    
    
    with open(file_specs_archs) as json_data:
        arch_specs_list = json.load(json_data)
    
    arch_specs_dict = {arch_specs_list[path_idx][arch_idx]['id']: arch_specs_list[path_idx][arch_idx]  for path_idx in range(len(arch_specs_list)) for arch_idx in range(len(arch_specs_list[path_idx]))}
        
    with open(file_test_results) as json_data:
        test_results_dict = json.load(json_data)
    
    
    ### iterate over architectures
    for arch_dir_i in dir_list_updated:
        
        local_id = arch_dir_i[len(arch_str_prefix):]
        local_id = int(local_id)
        
        assert arch_str_prefix + str(local_id) == arch_str_prefix + str(arch_specs_dict[local_id]['id'])
        
        empty = False
        try:
            pd_arch_i = pd.read_csv(path + arch_dir_i + '/metrics.csv')
        except pd.errors.EmptyDataError:
            empty = True
            cpt_empty_evals += 1
               
        if empty is False and pd_arch_i.empty is False:
            
            matrix_i = arch_specs_dict[local_id]['module_adjacency']
            list_ops_i = arch_specs_dict[local_id]['module_operations']
            hash_arch_i = arch_specs_dict[local_id]['unique_hash']
            num_params = test_results_dict[arch_dir_i]['num_params']

            dict_spec_mix_i = dict()
            dict_spec_mix_i['module_adjacency'] = matrix_i
            dict_spec_mix_i['module_operations'] = list_ops_i
            dict_spec_mix_i['trainable_parameters'] = num_params
            
            
            dict_arch_i_all_data_micro = dict()            
            dict_arch_i_all_data_macro = dict()            
            dict_arch_i_all_data_latency = dict() 
            dict_arch_i_all_data_MAC = dict() 
            
            for timestep_t in [107]: 
                
                dict_arch_i_perf_t_micro = dict()
                dict_arch_i_perf_t_macro = dict()
                dict_arch_i_perf_t_latency = dict() 
                dict_arch_i_perf_t_MAC = dict() 
                
                for metric_i in ["avg_macro", "avg_micro", 'inference', 'MACs']:
             
                    # store it
                    if 'macro' in metric_i:
                        validation_i_acc_t = pd_arch_i['validation_' + metric_i + '_accuracy'][timestep_t * 2]
                        training_i_acc_t = pd_arch_i['train_' + metric_i + '_accuracy'][timestep_t * 2 - 1]
                    
                        dict_arch_i_perf_t_macro['final_validation_accuracy'] = validation_i_acc_t
                        dict_arch_i_perf_t_macro['final_train_accuracy'] = training_i_acc_t
                        
                        if timestep_t == 107:
                            test_perf = test_results_dict[arch_dir_i]['accuracy']['macro']
                            dict_arch_i_perf_t_macro['final_test_accuracy'] = test_perf
                            
                            
                            # compute training time 
                            list_training_time = [pd_arch_i['training_time'][t_local * 2 - 1] for t_local in range(1, timestep_t + 2)]
                            #print(len(list_training_time))
                            dict_arch_i_perf_t_macro['final_training_time'] = sum(list_training_time)
                    
                    elif 'micro' in metric_i: 
                        validation_i_acc_t = pd_arch_i['validation_' + metric_i + '_accuracy'][timestep_t * 2]
                        training_i_acc_t = pd_arch_i['train_' + metric_i + '_accuracy'][timestep_t * 2 - 1]
                        
                        dict_arch_i_perf_t_micro['final_validation_accuracy'] = validation_i_acc_t
                        dict_arch_i_perf_t_micro['final_train_accuracy'] = training_i_acc_t
                        
                        if timestep_t == 107:
                            test_perf = test_results_dict[arch_dir_i]['accuracy']['micro']
                            dict_arch_i_perf_t_micro['final_test_accuracy'] = test_perf
                            
                            # compute training time 
                            list_training_time = [pd_arch_i['training_time'][t_local * 2 - 1] for t_local in range(1, timestep_t + 2 )]
                            #print(len(list_training_time))
                            dict_arch_i_perf_t_micro['final_training_time'] = sum(list_training_time)
                    elif 'inference' in metric_i: 
                        if timestep_t == 107:
                            dict_arch_i_perf_t_latency['final_validation_accuracy'] = test_results_dict[arch_dir_i]['avg_inference_time']
                            dict_arch_i_perf_t_latency['final_train_accuracy'] = test_results_dict[arch_dir_i]['avg_inference_time']
                            dict_arch_i_perf_t_latency['final_test_accuracy'] = test_results_dict[arch_dir_i]['avg_inference_time']
                            
                            # compute training time 
                            list_training_time = [pd_arch_i['training_time'][t_local * 2 - 1] for t_local in range(1, timestep_t + 2)]
                            #print(len(list_training_time))
                            dict_arch_i_perf_t_latency['final_training_time'] = sum(list_training_time)
                        
                    elif 'MACs' in metric_i:
                        if timestep_t == 107:
                            MAC_i = test_results_dict[arch_dir_i]['MACs']
                            MAC_i = float(MAC_i[:-1]) # "5.544M" -> 5.554
                            dict_arch_i_perf_t_MAC['final_validation_accuracy'] = MAC_i
                            dict_arch_i_perf_t_MAC['final_train_accuracy'] = MAC_i
                            dict_arch_i_perf_t_MAC['final_test_accuracy'] = MAC_i
                            
                            # compute training time 
                            list_training_time = [pd_arch_i['training_time'][t_local * 2 - 1] for t_local in range(1, timestep_t + 2)]
                            #print(len(list_training_time))
                            dict_arch_i_perf_t_MAC['final_training_time'] = sum(list_training_time)
                    else:
                        pass
            
                dict_arch_i_all_data_micro[ timestep_t + 1] = [dict_arch_i_perf_t_micro]
                dict_arch_i_all_data_macro[ timestep_t + 1] = [dict_arch_i_perf_t_macro]
                dict_arch_i_all_data_latency[ timestep_t + 1] = [dict_arch_i_perf_t_latency]
                dict_arch_i_all_data_MAC[ timestep_t + 1] = [dict_arch_i_perf_t_MAC]
                
            
            dict_database_to_pickle_micro[hash_arch_i] = (dict_spec_mix_i, dict_arch_i_all_data_micro)
            dict_database_to_pickle_macro[hash_arch_i] = (dict_spec_mix_i, dict_arch_i_all_data_macro)
            dict_database_to_pickle_latency[hash_arch_i] = (dict_spec_mix_i, dict_arch_i_all_data_latency)
            dict_database_to_pickle_MAC[hash_arch_i] = (dict_spec_mix_i, dict_arch_i_all_data_MAC)
            

    list_data_to_save = [dict_database_to_pickle_micro, dict_database_to_pickle_macro,
                            dict_database_to_pickle_latency, dict_database_to_pickle_MAC]

    list_filenames_to_save = [pickle_file_for_database_micro, pickle_file_for_database_macro, 
                                pickle_file_for_database_latency, pickle_file_for_database_MAC]

    
    print ('DONE!')
    