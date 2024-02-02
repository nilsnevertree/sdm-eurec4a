import numpy as np


def linear_func(x: np.ndarray, f_0: float = 2, slope: float = 1):
    """
    Linear function.

    :math:`y = slope * x + f_0`
    """
    return slope * x + f_0


def split_linear_func(
    x: np.ndarray, f_0: float = 2, slope_1: float = 1, slope_2: float = 2, x_split: float = 800
):
    """
    Split the array x into two arrays at the point x_split. The function is the
    concatenation of two linear functions with different slopes.

    :math:`y_1 = slope_1 * x + f_0` for x <= x_split
    :math:`y_2 = slope_2 * x + f_0 + (slope_1 - slope_2) * x_split` for x > x_split

    Parameters
    ----------
    x : np.ndarray
        The input array
    f_0 : float, optional
        The y-intercept, by default 2
    slope_1 : float, optional
        The slope of the first linear function, by default 1
    slope_2 : float, optional
        The slope of the second linear function, by default 2
    x_split : float, optional
        The x value at which the array is split, by default 800

    Returns
    -------
    np.ndarray
        The sum of the two linear functions

    Examples
    --------
    >>> x = np.arange(0, 1000, 100)
    >>> split_linear(x, f_0=2, slope_1=1, slope_2=2, x_split=800)
    array([  2., 102., 202., 302., 402., 502., 602., 702., 802., 902.])
    """
    x_1 = np.where(x <= x_split, x, np.nan)
    x_2 = np.where(x > x_split, x, np.nan)

    y_1 = linear_func(x=x_1, f_0=f_0, slope=slope_1)
    y_2 = linear_func(x=x_2, f_0=f_0 + (slope_1 - slope_2) * x_split, slope=slope_2)

    y_1 = np.where(x > x_split, 0, y_1)
    y_2 = np.where(x <= x_split, 0, y_2)
    return y_1 + y_2
