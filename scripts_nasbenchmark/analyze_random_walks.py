#!/usr/bin/env
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import statsmodels.api as sm


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
    plt.savefig(prefix_saving_location + 'random_walk_analysis/random_walk_id_' + str(walk_identifier) + '_' + str(timestep_training_local) + '_' + metric_name_local + '_.png') 
    plt.clf()

def plot_distribution_ruggedness(list_ruggedness_local, timestep_training_local, metric_name_local, prefix_saving_location, min_rw_length_local=-10):
    plt.title("PDF of Ruggedness for walks of minimal length - N=" + str(min_rw_length_local))
    plt.xlim([-20, 20])
    plt.hist(list_ruggedness_local, density=True, label='Training time (Epochs):' + str(timestep_training_local), 
         histtype='step', alpha=0.55, color='blue', bins=100)
    plt.ylabel("Density")
    plt.xlabel("Ruggedness")
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(prefix_saving_location + 'random_walk_analysis/distributions_of_ruggedness_' + str(timestep_training_local) + '_' + metric_name_local + '_.png')   
    plt.clf()    

def retrieve_performances(architecture_id, local_path, timestep_local):
    empty = False
    cpt_empty_evals_local = 0
    training_t_acc_t_local, validation_acc_t_local = 0.0, 0.0
    # collect validation and training architecture at time step t_j
    try:
        pd_arch_i_local = pd.read_csv(local_path + architecture_id + '/version_0/metrics.csv')
    except OSError:
        empty = True
        cpt_empty_evals_local += 1
        #print ("empty file: ", local_path + architecture_id + '/version_0/metrics.csv')
    except pd.errors.EmptyDataError:
        empty = True
        cpt_empty_evals_local += 1
        #print ("empty file: ", local_path + architecture_id + '/version_0/metrics.csv')

    if empty is False and pd_arch_i_local.empty is False:
        validation_acc_t_local = pd_arch_i_local['validation_' + metric_i + '_accuracy'][timestep_local * 2]
        training_t_acc_t_local = pd_arch_i_local['train_' + metric_i + '_accuracy'][timestep_local * 2 - 1]

    return training_t_acc_t_local, validation_acc_t_local, cpt_empty_evals_local

if __name__ == "__main__":
    min_rw_length = 6
    json_file = 'sequences.json'
    prefix_saving_location = './random_walk_analysis/'
    with open(json_file) as json_data:
        data = json.load(json_data)
    
    list_of_sequence_lengths = list()
    largest_rw = list()
    
    for k in data.keys():
        local_walk_length = len(data[k])
        list_of_sequence_lengths.append(local_walk_length)
        if local_walk_length >= min_rw_length:
            largest_rw.append((k, local_walk_length))
    
    plt.title("PDF of Random Walk lengths")
    plt.hist(list_of_sequence_lengths, density=True, label='PDF', 
         histtype='step', alpha=0.55, color='blue', bins=40)
    plt.ylabel("Density")
    plt.xlabel("Number of steps")
    #plt.legend(loc='upper right')
    plt.show()
    plt.savefig(prefix_saving_location + 'random_walk_analysis/distributions_of_steps_in_all_walks.png')    
    plt.clf()
    #print ('DONE!')
    
    
    path = '/p/project/hai_nasb_eo/training/logs/'

    for timestep in [4, 12, 36, 107]:    
        for metric_i in ["avg_macro", "avg_micro"]:

            list_of_walk_fitness = list()
            cpt_empty_evals = 0
            ### iterate over architectures
            for (arch_dir_i, random_walk_length_i) in largest_rw:
                #print((arch_dir_i, random_walk_length_i))
                local_list_validation_t = list()
                _, val_eval_i, cpt_empty_i = retrieve_performances(arch_dir_i, path, timestep)            
                local_list_validation_t.append(val_eval_i)
                cpt_empty_evals += cpt_empty_i
                
                if True: #cpt_empty_i == 0:
                    for j in range(random_walk_length_i):
                        local_step_code = str(j+1) + '_step'
                        #print(random_walk_length_i, data[arch_dir_i].keys())
                        local_step_architecture_id = data[arch_dir_i][local_step_code]['arch_code']
                        _, val_eval_j, cpt_empty_j = retrieve_performances(local_step_architecture_id, path, timestep)       
                        local_list_validation_t.append(val_eval_j)
                        cpt_empty_evals += cpt_empty_i
                        #print(local_step_architecture_id, train_eval_i)
                list_of_walk_fitness.append(local_list_validation_t)
            #print(list_of_walk_fitness)
    
            distribution_of_ruggedness = list()


            for id_walk, walk_i in enumerate(list_of_walk_fitness):
                plot_random_walk_i(walk_data=walk_i, walk_identifier=id_walk, 
                                   timestep_training_local=timestep, 
                                   metric_name_local=metric_i)        

                local_ruggedness = ruggedness(walk_i)
                distribution_of_ruggedness.append(local_ruggedness)

            plot_distribution_ruggedness(list_ruggedness_local=distribution_of_ruggedness, 
                                         timestep_training_local=timestep, 
                                         min_rw_length_local=min_rw_length,
                                         metric_name_local=metric_i)        
            
    print ('DONE!')
            