import numpy as np

from sdm_eurec4a.input_processing.models import linear_func, split_linear_func


def test_linear_func():
    """
    Test the linear_func function.

    This also includes np.nan values
    """

    x = np.arange(0, 11, 1, dtype=float)
    x[5] = np.nan
    should = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, np.nan, 14.0, 16.0, 18.0, 20.0, 22.0])
    result = linear_func(x, f_0=2, slope=2)
    np.testing.assert_array_equal(result, should)


def test_split_linear_func():
    """
    Test the split_linear_func function.

    This also includes np.nan values
    """

    x = np.arange(0, 11, 1, dtype=float)
    x[5] = np.nan
    should = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 9.0, 11.0, 13.0, 15.0, 17.0])
    result = split_linear_func(x, f_0=2, slope_1=1, slope_2=2, x_split=5)
    np.testing.assert_array_equal(result, should)
