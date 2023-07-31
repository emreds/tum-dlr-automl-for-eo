import os
import pickle
from pathlib import Path
import json
import yaml
import numpy as np
from json import JSONEncoder 
import shutil

def get_base_arch_paths(arch_folder: Path):
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

def get_checkpoint_paths(log_dir, exp_number="0_0", epoch="107", filter=set()) -> dict:
    """
    Returns the checkpoints for the given experiment number and epoch.

    Args:
        log_dir (_type_): _description_
        exp_number (str, optional): _description_. Defaults to "0_0".
        epoch (str, optional): _description_. Defaults to "107".
        filter (set, optional): Only returns the paths which has the `arch_code` in filter. Defaults to set().

    Returns:
        _type_: _description_
    """
    
    train_logs = os.listdir(log_dir)
    epoch_prefix = "epoch=" + epoch
    
    arch_checkpoint = {}
    for dir_name in train_logs:
        word_list = dir_name.split("_")
        # Network checkpoints are saved in format `arch_x_arch_x`.
        if len(word_list) > 2:
            arch_code = word_list[0] + '_' + word_list[1]
            if filter and arch_code not in filter:
                continue
            checkpoint_dir = log_dir / dir_name / exp_number / "checkpoints"
            epoch = [epoch_path for epoch_path in os.listdir(checkpoint_dir) if epoch_path.split("-")[0] == epoch_prefix]
            if epoch:
                arch_checkpoint[arch_code] = checkpoint_dir / epoch[0]
            else:
                continue
            
    return arch_checkpoint

def move_contents(source_dir, destination_dir):
    # Iterate over all files and subdirectories in the source directory
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)

        # Move the item to the destination directory
        shutil.move(item_path, destination_path)
    
        

def get_checkpoint_paths_clean(log_dir, epoch="107", filter=set()) -> dict:
    """
    Returns the checkpoints for the given experiment number and epoch.

    Args:
        log_dir (_type_): _description_
        exp_number (str, optional): _description_. Defaults to "0_0".
        epoch (str, optional): _description_. Defaults to "107".
        filter (set, optional): Only returns the paths which has the `arch_code` in filter. Defaults to set().

    Returns:
        _type_: _description_
    """
    
    train_logs = os.listdir(log_dir)
    epoch_prefix = "epoch=" + epoch
    
    arch_checkpoint = {}
    print(len(train_logs))
    for dir_name in train_logs:
        # Network checkpoints are saved in format `arch_x_arch_x`.
        
        arch_code = dir_name
        if filter and arch_code not in filter:
            continue
        epochs_dir = log_dir / dir_name / "epochs"
        #checkpoint_dir = log_dir / dir_name / "epochs" / "checkpoints"
            
        trained_epoch = [epoch_path for epoch_path in os.listdir(epochs_dir) if epoch_path.split("-")[0] == epoch_prefix]
        
        if trained_epoch:
            if arch_code in arch_checkpoint:
                print(f"Arch code already exists: {arch_code}")
            arch_checkpoint[arch_code] = epochs_dir / trained_epoch[0]
        else:
            continue
        
    return arch_checkpoint


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pickle_data = pickle.load(f)

    return pickle_data

def load_json(json_path):
    with open(json_path) as f:
        py_json = json.load(f)

    return py_json

def get_general_config():
    with open("../configs/general.yml", 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    return yaml_data

class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            return super(NumpyArrayEncoder, self).default(obj)