import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf


def mean_accuracy(acc_list):
    return float(np.mean(acc_list))


def variance(acc_list):
    return float(np.var(acc_list, ddof=1))


def positive_persistence(acc_list):
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[-top25_len:]
    current_top25 = np.argsort(acc_list[:, 1])[-top25_len:]
    # check the first two epochs, find common models
    initial_common_models = previous_top25 == current_top25

    if not any(initial_common_models):
        return 0

    previous_common_models = initial_common_models
    previous_top25 = current_top25

    for epoch in range(2, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[-top25_len:]
        # check the consecutive two epochs, find common models
        current_common_models = current_top25 == previous_top25
        # compare the current common models with previous commom models
        current_common_models = (
            (previous_common_models == True)
            & (current_common_models == True)
            & (previous_common_models == current_common_models)
        )

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return len(np.where(current_common_models == True)[0]) / len(
        np.where(initial_common_models == True)[0]
    )


def negative_persistence(acc_list):
    number_of_models, number_of_epochs = np.shape(acc_list)
    top25_len = number_of_models / 4
    top25_len = int(np.ceil(top25_len))
    previous_top25 = np.argsort(acc_list[:, 0])[:top25_len]
    current_top25 = np.argsort(acc_list[:, 1])[:top25_len]
    initial_common_models = previous_top25 == current_top25

    if not any(initial_common_models):
        return 0

    previous_common_models = initial_common_models
    previous_top25 = current_top25

    for epoch in range(2, number_of_epochs):
        current_top25 = np.argsort(acc_list[:, epoch])[:top25_len]
        current_common_models = current_top25 == previous_top25
        current_common_models = (
            (previous_common_models == True)
            & (current_common_models == True)
            & (previous_common_models == current_common_models)
        )

        previous_common_models = current_common_models
        previous_top25 = current_top25

    return len(np.where(current_common_models == True)[0]) / len(
        np.where(initial_common_models == True)[0]
    )


def ruggedness(data, lags=1):
    """
    Calculates ruggedness metric and returns the result.
    By definition calculates the Autocorrelation and takes the inverse of it.

    Args:
        data (np.array): Data to calculate ruggedness.
        lags (int): Number of lags to calculate.

    Returns:
        np.array
    """
    acorr = sm.tsa.acf(data, nlags=lags)
    rugs = 1 / acorr

    return rugs


def plot_ruggedness(data, exp_name, lags=1):
    """
    Plots the ruggedness.

    Args:
        data (np.array): Data to calculate autocorrelation.
        exp_name (string): Experiment name.
        lags (int): Number of lags to calculate.
    """
    rugs = ruggedness(data=data, lags=lags)
    exp_name += "_ruggedness"
    fig_name = get_fig_name(exp_name=exp_name, lags=lags)
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../reports/figures", fig_name)
    )
    plt.plot(rugs)
    plt.xticks(range(0, lags))
    plt.savefig(path)

    pass


def get_fig_name(exp_name, **kwargs):
    """
    Creates a unique figure name based on exp_name, parameters and datetime.

    Args:
        exp_name (string): Experiment name.

    Returns:
        string: Figure name.
    """

    fig_date = datetime.now().strftime("%Y%m%d%H%M%S")
    fig_name = f"{exp_name}_{fig_date}"
    if kwargs:
        for param, value in kwargs.items():
            fig_name += "_" + str(param) + "_" + str(value)

    fig_name += ".png"

    return fig_name
