#!/usr/bin/env
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 14})

def ruggedness(data, lags=1):
    """
    Calculates ruggedness metric and returns the result.
    By definition calculates the Autocorrelation and takes the inverse of it.
    Args:
        data (np.array): Data to calculate ruggedness.
        lags (int): Number of lags to calculate.
    Returns:
        np.array
    """
    acorr = sm.tsa.acf(data, nlags=lags)[1:].mean()
    rugs = 1 / acorr

    return rugs

def plot_random_walk_i(walk_data, walk_identifier, timestep_training_local, metric_name_local, prefix_saving_location):
    plt.title("Visualizing random Walk: ID " + str(walk_identifier) + ' - after Training time (Epochs):' + str(timestep_training_local))
    plt.plot(range(len(walk_data)), walk_data)
    plt.ylabel("Fitness")
    plt.xlabel("Number of steps")
    #plt.legend(loc='upper right')
    plt.show()
    plt.savefig(prefix_saving_location + 'random_walk_id_' + str(walk_identifier) + '_' + str(timestep_training_local) + '_' + metric_name_local + '_.png') 
    plt.clf()

def plot_distribution_ruggedness(list_ruggedness_local, timestep_training_local, metric_name_local, prefix_saving_location, min_rw_length_local=-10):
    if metric_name_local == "avg_macro":
        plt.title("PDF of Ruggedness - macro accuracy")
    else:
        plt.title("PDF of Ruggedness - micro accuracy")
    #plt.title("PDF of Ruggedness - ")
    plt.xlim([-20, 20])
    plt.hist(list_ruggedness_local, density=True, label='Training time (Epochs):' + str(timestep_training_local), 
         histtype='step', alpha=0.55, color='blue', bins=100)
    plt.ylabel("Density")
    plt.xlabel("Ruggedness")
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(prefix_saving_location + 'distributions_of_ruggedness_' + str(timestep_training_local) + '_' + metric_name_local + '_.png')   
    plt.clf()    

def retrieve_performances(architecture_id, local_path, timestep_local):
    empty = False
    cpt_empty_evals_local = 0
    training_t_acc_t_local, validation_acc_t_local = 0.0, 0.0
    # collect validation and training architecture at time step t_j
    architecture_folder = "arch_" + str(architecture_id)
    try:
        pd_arch_i_local = pd.read_csv(local_path + architecture_folder + '/metrics.csv')
    except OSError:
        empty = True
        cpt_empty_evals_local += 1
        print ("empty file: ", local_path + architecture_folder + '/metrics.csv')
    except pd.errors.EmptyDataError:
        empty = True
        cpt_empty_evals_local += 1
        print ("empty file: ", local_path + architecture_folder + '/metrics.csv')

    if empty is False and pd_arch_i_local.empty is False:
        validation_acc_t_local = pd_arch_i_local['validation_' + metric_i + '_accuracy'][timestep_local * 2]
        training_t_acc_t_local = pd_arch_i_local['train_' + metric_i + '_accuracy'][timestep_local * 2 - 1]

    return training_t_acc_t_local, validation_acc_t_local, cpt_empty_evals_local

if __name__ == "__main__":
    min_rw_length = 15
    json_file = '../nasbench_database/path_logs.json'
    prefix_saving_location = '../random_walk_analysis/'
    with open(json_file) as json_data:
        data = json.load(json_data)
    
    list_of_sequence_lengths = list()
    largest_rw = list()
    
    for i in range(len(data)):
        local_walk_length = len(data[i])
        list_of_sequence_lengths.append(local_walk_length)
        largest_rw.append((i, local_walk_length))
    
    plt.title("PDF of Random Walk lengths")
    plt.hist(list_of_sequence_lengths, density=True, label='PDF', 
         histtype='step', alpha=0.55, color='blue', bins=40)
    plt.ylabel("Density")
    plt.xlabel("Number of steps")
    plt.show()
    plt.savefig(prefix_saving_location + 'distributions_of_steps_in_all_walks.png')    
    plt.clf()
    
    path = '/p/project/hai_nasb_eo/sampled_paths/all_trained_archs/'

    for timestep in [4, 12, 36, 107]:    
        for metric_i in ["avg_macro", "avg_micro"]:

            list_of_walk_fitness = list()
            cpt_empty_evals = 0
            ### iterate over architectures
            for (arch_dir_i, random_walk) in enumerate(data):
                local_list_validation_t = list()
                
                _, val_eval_i, cpt_empty_i = retrieve_performances(random_walk[0]['id'], path, timestep)            
                local_list_validation_t.append(val_eval_i)
                cpt_empty_evals += cpt_empty_i
                
                for j in range(min_rw_length):
                    local_step_code = str(j+1) + '_step'
                    local_step_architecture_id = data[arch_dir_i][j]['id']
                    _, val_eval_j, cpt_empty_j = retrieve_performances(local_step_architecture_id, path, timestep)       
                    local_list_validation_t.append(val_eval_j)
                    cpt_empty_evals += cpt_empty_i
                    
                list_of_walk_fitness.append(local_list_validation_t)
    
            distribution_of_ruggedness = list()


            for id_walk, walk_i in enumerate(list_of_walk_fitness):
                plot_random_walk_i(walk_data=walk_i, walk_identifier=id_walk, 
                                   timestep_training_local=timestep, 
                                   metric_name_local=metric_i,
                                   prefix_saving_location=prefix_saving_location)        

                local_ruggedness = ruggedness(walk_i)
                distribution_of_ruggedness.append(local_ruggedness)

            plot_distribution_ruggedness(list_ruggedness_local=distribution_of_ruggedness, 
                                         timestep_training_local=timestep, 
                                         min_rw_length_local=min_rw_length,
                                         metric_name_local=metric_i,
                                         prefix_saving_location=prefix_saving_location)        
            
    print ('DONE!')
            