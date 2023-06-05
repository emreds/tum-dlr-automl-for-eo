#!/bin/bash -x

#SBATCH --account=hai_nasb_eo
#SBATCH --partition=develbooster
#SBATCH --job-name=eo_nas
#SBATCH --nodes=1     
#SBATCH --time=1:00:00  
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j

# go to the repository directory                                                                                                                                                                                   
cd /p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/

# connect interactively to the compute node for experiments
srun /p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/scripts/bash_slurm/inference_metrics.sh