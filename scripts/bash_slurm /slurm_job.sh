#!/bin/bash -x
​
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --account=hai_automleo
#SBATCH --time=00:10:00
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBTACH --gres=gpu:4
​
# go to the repository directory                                                                                                                                                                                   
cd /p/project/hai_nasb_eo/tum-dlr-automl-for-eo/
​
​
# Allocate a compute node for 30 minutes 
# on the develbooster partition, 
# with 4 GPUs, 
# and using the account of the project hai_automleo
​
# connect interactively to the compute node for experiments
srun --cpu_bind=none --nodes=1 --pty ./scripts/bash script_experiments.sh