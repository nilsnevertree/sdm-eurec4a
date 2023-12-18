import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scipy import signal
from sdm_eurec4a import statistics


def test_RMSE():
    # Example 1: Test with arrays of equal length
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    np.testing.assert_equal(statistics.RMSE(x, y), np.sqrt(14 / 3))

    # Example 2: Test with arrays of different length
    with pytest.raises(ValueError):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6])
        statistics.RMSE(x, y)

    # Example 3: Test with arrays include nan
    x = np.array([1, 2, np.nan])
    y = np.array([2, 4, 6])
    assert np.isnan(
        statistics.RMSE(x, y),
    )
