#!/bin/bash -x

#SBATCH --account=hai_nasb_eo
#SBATCH --partition=booster # booster or develbooster number of gpus per node
#SBATCH --gres=gpu:4   
#SBATCH --job-name=eo_nas  
#SBATCH --nodes=1                # node count
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)   
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j

# go to the repository directory                                                                                                                                                                                   
cd /p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/

# connect interactively to the compute node for experiments
srun /p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/scripts/bash_slurm/script_experiments.sh ${1}