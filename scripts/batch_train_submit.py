import json
import logging
import os
import subprocess

from pathlib import Path
from tum_dlr_automl_for_eo.utils import file

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(filename='train_lhc_randomwalk.log', level=logging.DEBUG, format=FORMAT)

ARCHITECTURE_FOLDER = "/p/project/hai_nasb_eo/training/sampled_archs"
SLURM_SCRIPT_PATH = "/p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/scripts/bash_slurm/slurm_job.sh"

def trigger_job(arch_path):
    """
    Triggers a slurm job for given architecture.

    Args:
        arch_path (str)
        
    Returns:
        job_id(str): Slurm Job id.
    """
    command = "sbatch" + ' ' + SLURM_SCRIPT_PATH +  ' ' + arch_path
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    error = result.stderr.decode('utf-8')
    output = result.stdout.decode('utf-8')
    job_id = None
    # Sometimes warnings are considered as error, so instead of cancelling the job, I just parse the warning.
    if error:
        logging.error(f"An error occurred in the submission of SLURM job.\n Error: {error}")
    if output: 
        logging.info(f"SLURM job submitted successfully.\n Output:{output}")
        job_id = output.strip().split(' ')[-1]
    
    return job_id


def get_slurm_output(user):
    """
    Checks the slurm job output for given user.

    Args:
        user (str): Slurm user name.

    Returns:
        data(List[Dict]): Slurm output in dictionary list format. 
    """
    command = "sacct" + ' ' + '-u'  + ' ' + user
    data = []
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    error = result.stderr.decode('utf-8')
    if error:
        logging.error(f"Error occurred while reading the SLURM job status.\n Error: {error}")
    else:
        logging.info("Reading SLURM job status is successful.")
        output = result.stdout.decode('utf-8')
        lines = output.strip().split("\n")
        #extract the column names
        columns = lines[0].split()
        #skip the line with dashes
        lines = lines[1:]
        for line in lines[2:]:
            values = line.split()
            #zip column names and values and create a dictionary
            row = dict(zip(columns, values))
            data.append(row)

    return data


if __name__ == "__main__": 
    # Cancel all jobs of a user, `scancel -u`
    arch_folder = Path(ARCHITECTURE_FOLDER)
    # We remove the `arch_specs.json` file using `[:-1]`
    arch_paths = sorted([str(path) for path in file.get_base_arch_paths(arch_folder)])[:-1]
    
    # I am submitting first 200 for first trial.
    # We have already trained the first two architectures for a trial
    arch_paths = arch_paths[600:1000]
    print(arch_paths)
    
    job_ids = [trigger_job(path) for path in arch_paths]
    id_arch = dict(zip(job_ids, arch_paths))

    #print(id_arch)
    #sample = arch_paths[-2]
    #job_id = trigger_job(sample)
    #output = get_slurm_output()
    #for row in output:
    #    if row["JobName"] == "torch-test":
    #        print(row["JobID"])
    #trigger_job(sample)
    with open("./job_ids_600-1000.json", 'w') as f:
        f.write(json.dumps(id_arch))