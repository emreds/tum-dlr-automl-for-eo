#!/bin/bash
rm output.txt

# create the required directory for data storage on the temporary but super fast file tmpfs filesystem
mkdir -p /dev/shm/hai_nasb_eo/data

# Copy the data to the target storage directory
#cp /p/project/hai_automleo/data/*h5 /dev/shm/traore1/hai_automleo/data/
cp /p/project/hai_nasb_eo/emre/data/*h5 /dev/shm/hai_nasb_eo/data

# Load the modules and libraries for the experiments
module load Stages/2020  GCCcore/.9.3.0 Singularity-Tools/2020-Python-3.7.9

# Run the container with support for nvidia and cuda
# apptainer exec --nv /p/project/hai_automleo/utils_hpo_pop_est/docker_images/hpo_pop_estimation_version1.sif bash mini_script_local_commands.sh
#!!!!! we will add our container trigger here.