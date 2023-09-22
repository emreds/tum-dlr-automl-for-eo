from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule
from tum_dlr_automl_for_eo.datamodules.So2SatDataSet import So2SatDataSet


def plot_distribution(data: dict, title: str, save_path: str) -> None: 
    """
    Plots the distribution of the data.
    Args:
        data (dict): Data to plot.
    Returns:
        None
    """
    # Extract keys and values from the data dictionary
    x_values = list(data.keys())
    y_values = list(data.values())

    # Generate a list of different colors for each bar (random colors in this case)
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_values)))

    # Create the bar plot with different colors
    plt.bar(x_values, y_values, color=colors)

    # Set integer labels on the x-axis
    plt.xticks(x_values)
    
    # Set plot title and axes labels
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title(title)
    
    # Display the plot and save
    plt.show()
    plt.savefig(save_path)
    
    # Clear the plot
    plt.clf()

if __name__ == "__main__":

    data_path = "/p/project/hai_nasb_eo/data/"
    data_module = EODataModule(data_path, "Sentinel-2")
    save_path = "/p/project/hai_nasb_eo/emre/tum-dlr-automl-for-eo/sample_analysis_all"

    full_class_dist = defaultdict(int)
    train_class_dist = defaultdict(int)
    val_class_dist = defaultdict(int)
    test_class_dist = defaultdict(int)

    data_module.setup_training_data(None)

    for label in data_module.training_data.data['label']:
        full_class_dist[np.nonzero(label)[0][0]] += 1
        train_class_dist[np.nonzero(label)[0][0]] += 1

    data_module.setup_validation_data(None)

    for label in data_module.validation_data.data['label']:
        full_class_dist[np.nonzero(label)[0][0]] += 1
        val_class_dist[np.nonzero(label)[0][0]] += 1

    data_module.setup_testing_data(None)

    for label in data_module.testing_data.data['label']:
        full_class_dist[np.nonzero(label)[0][0]] += 1
        test_class_dist[np.nonzero(label)[0][0]] += 1
        
    plot_distribution(full_class_dist, "Full class distribution", save_path + "/full_class_dist.png")
    plot_distribution(train_class_dist, "Train class distribution", save_path + "/train_class_dist.png")
    plot_distribution(val_class_dist, "Validation class distribution", save_path + "/val_class_dist.png")
    plot_distribution(test_class_dist, "Test class distribution", save_path + "/test_class_dist.png")
    
