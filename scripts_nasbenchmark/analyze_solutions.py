#!/usr/bin/env
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    path = "/p/project/hai_nasb_eo/training/logs/arch_0/version_0/metrics.csv"
    path = '/p/project/hai_nasb_eo/sampled_paths/all_trained_archs/'

    ### collect all architectures ids in directory
    # list of all content in a directory, filtered so only directories are returned
    dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
    # deal with duplicates
    dir_list_updated = [dir_local for dir_local in dir_list if len(set(dir_local.split('_'))) == len(dir_local.split("_"))] 
    
    file_test_results = '../nasbench_database/test_results_sampled_all_paths.json'
    prefix_saving_location = '../sample_analysis/'
    N_bins = 25
    
    with open(file_test_results) as json_data:
        test_results_dict = json.load(json_data)
    
    for timestep in [4, 12, 36, 107]:    
        for metric_i in ["avg_macro", "avg_micro"]:

            list_training_t = list()
            list_validation_t = list()
            cpt_empty_evals = 0
            ### iterate over architectures
            for arch_dir_i in dir_list_updated:
                #print("HERE: ", path + arch_dir_i)
                empty = False
                # collect validation and training architecture at time step t_j
                try:
                    pd_arch_i = pd.read_csv(path + arch_dir_i + '/metrics.csv')
                except pd.errors.EmptyDataError:
                    empty = True
                    cpt_empty_evals += 1
                    print ("empty file: ", path + arch_dir_i + '/metrics.csv')

                if empty is False and pd_arch_i.empty is False:
                    validation_acc_t = pd_arch_i['validation_' + metric_i + '_accuracy'][timestep * 2]
                    training_t_acc_t = pd_arch_i['train_' + metric_i + '_accuracy'][timestep * 2 - 1]

                    # store it
                    list_training_t.append(training_t_acc_t)
                    list_validation_t.append(validation_acc_t)
            
            
            if timestep == 107:
                
                list_test_t = list()
                list_test_latency = list()
                mode_test = 'macro'
                
                if 'micro' in metric_i:
                    mode_test = 'micro'
                
                keys_architectures_results_test = list(test_results_dict.keys())
                
                for k_test_arch in keys_architectures_results_test:
                    #print(test_results_dict[k_test_arch])
                    test_result_k = test_results_dict[k_test_arch]['accuracy'][mode_test]
                    test_latency_k= test_results_dict[k_test_arch]['avg_inference_time']
                    
                    list_test_t.append(test_result_k)
                    list_test_latency.append(test_latency_k)
                    
                
                plt.hist(list_test_t, density=True, cumulative=False, label='Test', 
                 histtype='step', alpha=0.55, color='green', bins=N_bins)
                
            # plot the CDFs and PDFs of validation and training accuracies at timesteps t_js
            plt.hist(list_training_t, density=True, cumulative=False, label='Train', 
                 histtype='step', alpha=0.55, color='purple', bins=N_bins)

            plt.hist(list_validation_t, density=True, cumulative=False, label='Validation', 
                 histtype='step', alpha=0.55, color='red', bins=N_bins)
                                       
            plt.title("PDF of performance after Epoch :" + str(timestep) + '- Metric: ' + metric_i)
            plt.ylabel("Density")
            plt.xlabel("Fitness value (Accuracy)")
            plt.legend(loc='upper left')
            plt.show()
            plt.savefig(prefix_saving_location + 'results_PDF_' + str(timestep) + '_' + metric_i + '_' + '_.png')    
            plt.clf()

            plt.hist(list_training_t, density=True, cumulative=True, label='Train', 
                 histtype='step', alpha=0.55, color='purple', bins=N_bins)

            plt.hist(list_validation_t, density=True, cumulative=True, label='Validation', 
                 histtype='step', alpha=0.55, color='red', bins=N_bins)
            
            if timestep == 107:
                plt.hist(list_test_t, density=True, cumulative=True, label='Test', 
                     histtype='step', alpha=0.55, color='green', bins=N_bins)
                
            #plt.title("CDF of performance after Epoch:" + str(timestep) + '- Metric: ' + metric_i)
            plt.ylabel("Density")
            plt.xlabel("Fitness value (Accuracy)")
            plt.legend(loc='upper left')
            plt.show()
            plt.savefig(prefix_saving_location + 'results_CDF_' + str(timestep) + '_' + metric_i  + '_.png')    
            plt.clf()
            
            if timestep == 107:
                plt.hist(list_test_latency, density=True, cumulative=True, label='Test', 
                     histtype='step', alpha=0.55, color='blue', bins=N_bins)
                
                plt.ylabel("Density")
                plt.xlabel("Latency value (in Milliseconds)")
                plt.legend(loc='upper left')
                plt.show()
                plt.savefig(prefix_saving_location + 'latency_CDF_' + str(timestep) + '_' + metric_i + '_' + '_.png')    
                plt.clf()
                

                plt.hist(list_test_latency, density=True, cumulative=False, label='Test', 
                     histtype='step', alpha=0.55, color='blue', bins=N_bins)
                
                plt.ylabel("Density")
                plt.xlabel("Latency value (in Milliseconds)")
                plt.legend(loc='upper left')
                plt.show()
                plt.savefig(prefix_saving_location + 'latency_PDF_' + str(timestep) + '_' + metric_i + '_' + '_.png')    
                plt.clf()

            print("Number of items: ", len(list_validation_t), len(dir_list_updated))
            print ('DONE!')
            