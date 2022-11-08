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


def test_mean_accuracy():

    assert custom_metrics.mean_accuracy(dummy_acc_list) == 5.5
    assert custom_metrics.mean_accuracy(real_acc_list) == 92.26762751738231


def test_variance():

    assert custom_metrics.variance(dummy_acc_list) == 9.166666666666666
    assert custom_metrics.variance(real_acc_list) == 3.8413658161348487
