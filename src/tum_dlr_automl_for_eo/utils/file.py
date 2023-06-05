import os
from pathlib import Path


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

def get_checkpoint_paths(log_dir, exp_number="0_0", epoch="107") -> dict:
    """
    Returns the checkpoints for the given experiment number and epoch.

    Args:
        log_dir (_type_): _description_
        exp_number (str, optional): _description_. Defaults to "0_0".
        epoch (str, optional): _description_. Defaults to "107".

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
            checkpoint_dir = log_dir / dir_name / exp_number / "checkpoints"
            epoch = [epoch_path for epoch_path in os.listdir(checkpoint_dir) if epoch_path.split("-")[0] == epoch_prefix]
            if epoch:
                arch_checkpoint[arch_code] = checkpoint_dir / epoch[0]
            else:
                continue
            
    return arch_checkpoint