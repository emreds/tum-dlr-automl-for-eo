import glob
import os
from datetime import datetime
import random
import sys

import numpy as np
from . import helper_acc_list

from src.tum_dlr_automl_for_eo.utils import custom_metrics


dummy_acc_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
real_acc_list = [
    88.29126358032227,
    93.23918223381042,
    93.23918223381042,
    92.79847741127014,
    93.23918223381042,
    92.79847741127014,
]

dummy_2d_acc_list = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
real_2d_acc_list = np.array(
    [
        [88.29126358032227, 93.23918223381042],
        [93.23918223381042, 92.79847741127014],
        [93.23918223381042, 92.79847741127014],
    ]
)


def test_mean_accuracy():

    assert custom_metrics.mean_accuracy(dummy_acc_list) == 5.5
    assert custom_metrics.mean_accuracy(real_acc_list) == 92.26762751738231
    assert custom_metrics.mean_accuracy(dummy_2d_acc_list) == 5.5
    assert custom_metrics.mean_accuracy(real_2d_acc_list) == 92.26762751738231


def test_variance():

    assert custom_metrics.variance(dummy_acc_list) == 9.166666666666666
    assert custom_metrics.variance(real_acc_list) == 3.8413658161348487
    assert custom_metrics.variance(dummy_2d_acc_list) == 9.166666666666666
    assert custom_metrics.variance(real_2d_acc_list) == 3.8413658161348487

dummy_arc_list_0 = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]])

dummy_acc_list_0 = [92, 45, 67, 89, 97, 96, 39, 32, 76, 23, 48]
dummy_acc_list_1 = [87, 63]
dummy_arc_list_1 = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 1]])


def test_positive_persistence():
    assert custom_metrics.positive_persistence(helper_acc_list.random_acc_list) == [1.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert custom_metrics.positive_persistence(helper_acc_list.uniform_acc_list) == [1.0 for i in range(helper_acc_list.uniform_acc_list.shape[1])]



def test_negative_persistence():
    assert custom_metrics.negative_persistence(helper_acc_list.random_acc_list) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert custom_metrics.negative_persistence(helper_acc_list.uniform_acc_list) == [1.0 for i in range(helper_acc_list.uniform_acc_list.shape[1])]

def test_ruggedness():
    results_1_lag = np.array([1, -13.99825467])
    results_5_lag = np.array(
        [1, -13.99825467, -9.05220222, -29.49523707, -5.73765437, -9.09908685]
    )

    assert np.allclose(custom_metrics.ruggedness(real_acc_list), results_1_lag)
    assert np.allclose(custom_metrics.ruggedness(real_acc_list, lags=5), results_5_lag)


def test_get_fig_name():
    args = ["lorem", "ipsum", "dolor", "sit", "42"]
    fig_name = custom_metrics.get_fig_name(exp_name="lorem", ipsum="dolor", sit=42)
    for arg in args:
        assert arg in fig_name


def test_plot_ruggedness():
    current_time = datetime.now()
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../reports/figures")
    )
    custom_metrics.plot_ruggedness(data=real_acc_list, exp_name="exp_1")
    latest_file_name, latest_file_time = file_check_helper(path=path, time=current_time)
    assert "ruggedness" in latest_file_name
    assert latest_file_time.time() > current_time.time()


def file_check_helper(path, time):
    """
    Helper function to return latest modified file's
    name and modification time in a folder.

    Args:
        path (string): Path to folder.

    Returns:
        string, datetime: File name and datetime object.
    """
    latest_file_time = time
    latest_file_name = ""
    path = str(path) + "/*"
    for file in glob.iglob(path):
        file_time = datetime.fromtimestamp(os.path.getmtime(file))
        if file_time.time() > latest_file_time.time():
            latest_file_time = file_time
            latest_file_name = file

    return latest_file_name, latest_file_time


def test_local_optima():
    # This is a stochastic process, so the tests need to be done accordingly 
    number_of_starting_points = random.randint(1, 10)
    number_of_iters = 1
    result = custom_metrics.search_local_optima(dummy_arc_list_0, dummy_acc_list_0, number_of_starting_points, 5, 50,number_of_iters)
    assert len(result[0]) == number_of_iters and result[1] > 0
    result1 = custom_metrics.search_local_optima(dummy_arc_list_1, dummy_acc_list_1, 1, 8, 50, number_of_iters)
    result2 = custom_metrics.search_local_optima(dummy_arc_list_1, dummy_acc_list_1, 2, 8, 50, number_of_iters)
    assert len(result1[0]) == number_of_iters and result1[1] > 0
    assert len(result2[0]) == number_of_iters and result2[1] > 0