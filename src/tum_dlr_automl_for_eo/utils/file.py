import os
from pathlib import Path


def get_arch_paths(arch_folder: Path):
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