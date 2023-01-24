import json
import logging
import os
import subprocess

from pathlib import Path

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(filename='train_lhc_randomwalk.log', level=logging.DEBUG, format=FORMAT)

ARCHITECTURE_FOLDER = "/p/project/hai_nasb_eo/training/sampled_archs"
SLURM_SCRIPT_PATH = "./bash_slurm/slurm_job.sh"


def get_arch_paths(arch_folder):
    """
    Returns the list of absolute architecture paths.

    Args:
        arch_folder (Path): Architecture folder.

    Returns:
        List[Path]: List of architecture paths.
    """
    arch_paths = []
    arch_names = os.listdir(arch_folder)

    for name in arch_names:
        arch_paths.append(arch_folder / name)
        
    return arch_paths

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
    arch_paths = sorted([str(path) for path in get_arch_paths(arch_folder)])[:-1]

    # We just take 2 architectures for a trial
    arch_paths = arch_paths[:2]
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
    with open("./job_ids.json", 'w') as f:
        f.write(json.dumps(id_arch))