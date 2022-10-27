"""
    This class is only here to demonstrate testing usage.
"""


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case("semaphore") == "Semaphore"
