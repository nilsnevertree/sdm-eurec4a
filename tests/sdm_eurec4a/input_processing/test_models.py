import numpy as np

from sdm_eurec4a.input_processing.models import linear_func, split_linear_func

def test_linear_func():
    """
    Test the linear_func function. 
    This also includes np.nan values
    """
    
    x = np.arange(0, 11, 1, dtype = float)
    x[5] = np.nan
    should = np.asarray([ 2.,  4.,  6.,  8.,  10., np.nan,  14., 16., 18., 20., 22.])
    result = linear_func(x, f_0=2, slope=2)
    np.testing.assert_array_equal(result, should)

def test_split_linear_func():
    """
    Test the split_linear_func function. 
    This also includes np.nan values
    """
    
    x = np.arange(0, 11, 1, dtype = float)
    x[5] = np.nan
    should = np.asarray([ 2.,  3.,  4.,  5.,  6., np.nan,  9., 11., 13., 15., 17.])
    result = split_linear_func(x, f_0=2, slope_1=1, slope_2=2, x_split=5)
    np.testing.assert_array_equal(result, should)
