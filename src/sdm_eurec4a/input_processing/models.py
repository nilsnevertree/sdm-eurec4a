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


def lnnormaldist(
    radii: np.ndarray, scale_factors: float, geometric_means: float, geometric_sigmas: float
) -> np.ndarray:
    """
    Calculate probability of radii given the paramters of a lognormal distribution
    according to equation 5.8 of "An Introduction to clouds from the Microscale to
    Climate" by Lohmann, Luond and Mahrt.

    Note
    ----
    The parameters geometric_means and geometric_sigmas are the geometric mean and geometric
    standard deviation of the distribution, not the arithmetic mean and
    standard deviation.
    The scale in which radii is given, is the same as the scale in which
    geometric_means and geometric_sigmas needs to be given.
    The scale_factors is the total number of particles N_a in the distribution
    [#/m^3]

    Parameters
    ----------
    radii : array_like
        radii [m] to calculate probability for
    scale_factors : float
        scale factor for distribution (see eq. 5.2)
        It is the total number particles N_a in the distribution [#/m^3]
    geometric_means : float
        geometric mean of distribution (see eq. 5.5)
    geometric_sigmas : float
        geometric standard deviation of distribution (see eq. 5.6)

    Returns
    -------
    dn_dlnr : array_like
        probability of each radius in radii [m^-1]
    """

    sigtilda = np.log(geometric_sigmas)
    mutilda = np.log(geometric_means)

    norm = scale_factors / (np.sqrt(2 * np.pi) * sigtilda)
    exponent = -((np.log(radii) - mutilda) ** 2) / (2 * sigtilda**2)

    dn_dlnr = norm * np.exp(exponent)  # eq.5.8 [lohmann intro 2 clouds]

    return dn_dlnr
