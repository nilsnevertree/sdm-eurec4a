from typing import Tuple, Dict, Union, Callable, TypedDict
import numpy as np
import xarray as xr
from scipy.optimize import least_squares, Bounds
from inspect import signature


def linear_func(x: np.ndarray, f_0: float = 2, slope: float = 1):
    """
    Linear function.

    :math:`y = slope * x + f_0`
    """
    return slope * x + f_0


def split_linear_func(
    x: Union[np.ndarray, xr.DataArray],
    f_0: float = 2,
    slope_1: float = 1,
    slope_2: float = 2,
    x_split: float = 800,
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

    if isinstance(x, np.ndarray):

        x_1 = np.where(x <= x_split, x, np.nan)
        x_2 = np.where(x > x_split, x, np.nan)

        y_1 = linear_func(x=x_1, f_0=f_0, slope=slope_1)
        y_2 = linear_func(x=x_2, f_0=f_0 + (slope_1 - slope_2) * x_split, slope=slope_2)

        y = np.where(x <= x_split, y_1, y_2)
        return y
    elif isinstance(x, xr.DataArray):

        x_1 = x.where(x <= x_split)
        x_2 = x.where(x > x_split)

        y_1 = linear_func(x=x_1, f_0=f_0, slope=slope_1)
        y_2 = linear_func(x=x_2, f_0=f_0 + (slope_1 - slope_2) * x_split, slope=slope_2)

        y = xr.where(x <= x_split, y_1, y_2)
        return y


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


class LnParams(TypedDict):
    """
    A TypedDict for parameters of the log-normal distribution.

    Attributes:
        mu1 (float): Mean of the log-normal distribution.
        sigma1 (float): Standard deviation of the log-normal distribution.
        scale_factor1 (float): Scale factor for the log-normal distribution.
    """

    mu1: float
    sigma1: float
    scale_factor1: float


def standard_normal(x: np.ndarray) -> np.ndarray:
    """Returns probability distribution according to
    standard normal distribution

    The standard normal distribution is given by:
    n(x) = 1 / sqrt(2 * pi) * exp(-x^2 / 2)

    Parameters:
    -----------
    x : np.ndarray
        array of independent variable

    Returns:
    --------
    probs : np.ndarray
        probability distribution
    """

    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def normal_distribution(x: np.ndarray, mu: float, sigma: float, scale: float) -> np.ndarray:
    """calculate probability of independent variable `x` given the paramters of a
    normal dsitribution

    The general normal distribution is given by:
    g(x; mu, sig) = 1 / sig * n((x - mu) / sig)
    with n being the standard normal distribution.

    Parameters
    ----------
    x : np.ndarray
        array of independent variable
    mu : float
        mean of the distribution
    sig : float
        standard deviation of the distribution
    scale : float
        scale factor

    Returns
    -------
    result : np.ndarray
        probability distribution
    """
    z = (x - mu) / sigma
    result = 1 / sigma * standard_normal(z)
    return scale * result


def log_normal_distribution(x: np.ndarray, mu: float, sigma: float, scale: float) -> np.ndarray:
    """calculate probability of independent variable `x` given the paramters of a
    lognormal dsitribution

    The general lognormal distribution is given by:
    g(x; mu, sig) = scale * 1 / (x * sig) * n((ln(x) - mu) / sig)
    with n being the standard normal distribution.

    Parameters
    ----------
    x : np.ndarray
        array of independent variable
    mu : float
        mean of the distribution
    sig : float
        standard deviation of the distribution
    scale : float
        scale factor

    Returns
    -------
    result : np.ndarray
        probability distribution
    """
    z = (np.log(x) - mu) / sigma
    result = 1 / (x * sigma) * standard_normal(z)
    return scale * result


def log_normal_distribution_all(
    x: np.ndarray,
    mu: float,
    sigma: float,
    scale: float,
    parameter_space: str = "direct",
    x_space: str = "linear",
    density_scaled: bool = False,
) -> np.ndarray:
    """calculate probability of independent variable `x` given the paramters of a
    lognormal dsitribution.
    Here one assumes the geometric mean and standard deviation.
    The natural logarithm is used.

    The general lognormal distribution is given by:
    g(x; mu, sig) = 1 / (x * ln(sig)) * n((ln(x) - ln(mu)) / ln(sig))
    with n being the standard normal distribution.

    Parameters
    ----------
    x : np.ndarray
        array of indepedent variable
    mu : float
        geometric mean of the distribution
    sigma : float
        geometric standard deviation of the distribution
    scale : float
        scale factor
    parameter_space : str
        Defines the x_space in which the parameters are given.
        Default is 'direct'
        - If 'direct' (the default), it is assumed that the given ``mu`` and ``sigma`` are the parameters of the distribution.
        - If 'geometric', it is assumed that the given ``mu`` and ``sigma`` are the geometric mean and standard deviation.
        - If 'exact', it is assumed that the given ``mu`` and ``sigma`` are the exact mean and standard deviation.
        default is False
    x_space : str
        Defines the x_space in which the distribution is calculated.
        In other words in which x_space the independent variable is given.
        Default is 'linear'
        - If 'linear' (the default), the distribution is it assumed that x is given in linear x_space.
        So for instance radius in m.
        - If 'ln', the distribution it is assumed that x is given in natural logarithm x_space.
        So for instance radius in ln(m).
        - If 'cleo', the distribution is calculated in the linear x_space but the independent variable is multiplied by x.
    density_scaled : bool
        If True, the distribution is scaled to a density distribution.
        The integral over the given x values is 1 * scale.
        NOTE: That this is NOT a analytical solution! If you change the input of x, you have to rescale the distribution.
        Default is False.

    Returns
    -------
    result : np.ndarray
        probability distribution
    """

    if parameter_space == "direct":
        mu = mu
        sigma = sigma
    elif parameter_space == "geometric":
        mu = np.log(mu)
        sigma = np.log(sigma)
    else:
        raise NotImplementedError("The requested parameter x_space does not exist.")
    if x_space == "linear":
        # in the linear x_space, the distribution is calculated as usual
        # R(x) = scale * L(x; mu, sigma)
        result = log_normal_distribution(x, mu, sigma, 1)
    elif x_space == "ln":
        # in the ln x_space, the distribution is calculated as
        # R(x) = scale * x * L(exp(x); mu, sigma)
        result = x * log_normal_distribution(x, mu, sigma, 1)
    else:
        raise NotImplementedError("The requested x_space does not exist.")
    if density_scaled:
        result = result / np.nansum(result)
        return result * scale
    else:
        return scale * result


def ln_normal_distribution(
    t: np.ndarray,
    mu1: float,
    sigma1: float,
    scale_factor1: float,
) -> np.ndarray:
    """
    Compute the log-normal distribution.

    Parameters:
        t (np.ndarray): Independent variable.
        mu1 (float): Mean of the log-normal distribution.
        sigma1 (float): Standard deviation of the log-normal distribution.
        scale_factor1 (float): Scale factor for the log-normal distribution.

    Returns:
        np.ndarray: The computed log-normal distribution.
    """
    result = t * 0

    sigtilda = np.log(sigma1)
    mutilda = np.log(mu1)

    norm = scale_factor1 / (np.sqrt(2 * np.pi) * sigtilda)
    exponent = -((np.log(t) - mutilda) ** 2) / (2 * sigtilda**2)

    dn_dlnr = norm * np.exp(exponent)  # eq.5.8 [lohmann intro 2 clouds]

    result += dn_dlnr

    return result


class DoubleLnParams(TypedDict):
    """
    A TypedDict for parameters of the double log-normal distribution.

    Attributes:
        mu1 (float): Mean of the first log-normal distribution.
        sigma1 (float): Standard deviation of the first log-normal distribution.
        scale_factor1 (float): Scale factor for the first log-normal distribution.
        mu2 (float): Mean of the second log-normal distribution.
        sigma2 (float): Standard deviation of the second log-normal distribution.
        scale_factor2 (float): Scale factor for the second log-normal distribution.
    """

    mu1: float
    sigma1: float
    scale_factor1: float
    mu2: float
    sigma2: float
    scale_factor2: float


def double_log_normal_distribution(
    x: np.ndarray,
    mu1: float,
    sigma1: float,
    scale1: float,
    mu2: float,
    sigma2: float,
    scale2: float,
) -> np.ndarray:
    """calculate probability of independent variable `x` given the paramters of a
    lognormal dsitribution

    The general lognormal distribution is given by:
    g(x; mu1, sig1, mu2, sig2) = LnNormal(x; mu1, sig1) + LnNormal(x; mu2, sig2)
    with n being the standard normal distribution.

    Parameters
    ----------
    x : np.ndarray
        array of independent variable
    mu : float
        mean of the distribution
    sig : float
        standard deviation of the distribution
    scale : float
        scale factor

    Returns
    -------
    result : np.ndarray
        probability distribution
    """
    results = log_normal_distribution(x, mu1, sigma1, scale1) + log_normal_distribution(
        x, mu2, sigma2, scale2
    )
    return results


def double_log_normal_distribution_all(
    x: np.ndarray,
    mu1: float,
    sigma1: float,
    scale1: float,
    mu2: float,
    sigma2: float,
    scale2: float,
    parameter_space: str = "direct",
    x_space: str = "linear",
    density_scaled: bool = False,
) -> np.ndarray:
    """
    Compute the sum of two log-normal distributions.

    Parameters:
    -----------
    x (np.ndarray):
        Input array of values.
    mu1 (float):
        Mean of the first log-normal distribution.
    sigma1 (float):
        Standard deviation of the first log-normal distribution.
    scale1 (float):
        Scale factor for the first log-normal distribution.
    mu2 (float):
        Mean of the second log-normal distribution.
    sigma2 (float):
        Standard deviation of the second log-normal distribution.
    scale2 (float):
        Scale factor for the second log-normal distribution.
    parameter_space (str, optional):
        Parameter space to use, either "direct" or "geometric". Defaults to "direct".
    x_space (str, optional):
        Space of the input values, either "linear" or "log". Defaults to "linear".
    density_scaled (bool, optional):
        Whether to scale the density. Defaults to False.

    Returns:
    --------
    np.ndarray:
        The sum of the two log-normal distributions evaluated at `x`.
    """

    result1 = log_normal_distribution_all(
        x=x,
        mu=mu1,
        sigma=sigma1,
        scale=scale1,
        parameter_space=parameter_space,
        x_space=x_space,
        density_scaled=density_scaled,
    )
    result2 = log_normal_distribution_all(
        x=x,
        mu=mu2,
        sigma=sigma2,
        scale=scale2,
        parameter_space=parameter_space,
        x_space=x_space,
        density_scaled=density_scaled,
    )
    return result1 + result2


def double_ln_normal_distribution(
    t: Union[np.ndarray, xr.DataArray],
    mu1: Union[float, np.ndarray, xr.DataArray],
    sigma1: Union[float, np.ndarray, xr.DataArray],
    scale_factor1: Union[float, np.ndarray, xr.DataArray],
    mu2: Union[float, np.ndarray, xr.DataArray],
    sigma2: Union[float, np.ndarray, xr.DataArray],
    scale_factor2: Union[float, np.ndarray, xr.DataArray],
) -> Union[np.ndarray, xr.DataArray]:
    """
    Compute the double log-normal distribution
    based on the geometric means and standard deviations of the two distributions.

    Parameters:
    -----------
    t : Union[np.ndarray, xr.DataArray]
        Independent variable.
    mu1 : Union[float, np.ndarray, xr.DataArray]
        Geometric mean of the first log-normal distribution.
    sigma1 : Union[float, np.ndarray, xr.DataArray]
        Geometric standard deviation of the first log-normal distribution.
    scale_factor1 : Union[float, np.ndarray, xr.DataArray]
        Scale factor for the first log-normal distribution.
    mu2 : Union[float, np.ndarray, xr.DataArray]
        Geometric mean of the second log-normal distribution.
    sigma2 : Union[float, np.ndarray, xr.DataArray]
        Geometric standard deviation of the second log-normal distribution.
    scale_factor2 : Union[float, np.ndarray, xr.DataArray]
        Scale factor for the second log-normal distribution.

    Returns:
    --------
    Union[np.ndarray, xr.DataArray]
        The computed double log-normal
    """
    result = t * 0

    for mu, sigma, scale_factor in zip(
        (mu1, mu2),
        (sigma1, sigma2),
        (scale_factor1, scale_factor2),
    ):
        sigtilda = np.log(sigma)
        mutilda = np.log(mu)

        norm = scale_factor / (np.sqrt(2 * np.pi) * sigtilda)
        exponent = -((np.log(t) - mutilda) ** 2) / (2 * sigtilda**2)

        dn_dlnr = norm * np.exp(exponent)  # eq.5.8 [lohmann intro 2 clouds]

        result = result + dn_dlnr

    return result


class SaturatedLinearParams(TypedDict):
    """
    A TypedDict for parameters of the saturated linear function.

    Attributes:
        f_0 (float): The y-intercept.
        slope_1 (float): The slope of the first linear function.
        saturation_value (float): The value at which the function saturates.
    """

    f_0: float
    slope_1: float
    saturation_value: float


def saturated_linear_func(
    x: Union[np.ndarray, xr.DataArray], f_0: float = 2, slope_1: float = 1, saturation_value: float = 1
):
    """
    This function is a linear function that saturates at a certain value.
    Above this value, the function is constant.

    Parameters
    ----------
    x : np.ndarray
        The input array
    f_0 : float, optional
        The y-intercept, by default 2
    slope_1 : float, optional
        The slope of the linear function, by default 1
    saturation_value : float, optional
        The value at which the function saturates, by default 1

    Returns
    -------
    np.ndarray
        The computed saturated linear function
    """

    x_split = (saturation_value - f_0) / slope_1
    return split_linear_func(x=x, f_0=f_0, slope_1=slope_1, slope_2=0, x_split=x_split)


def create_variance_field(
    y: np.ndarray,
    variance: Union[None, float, int, np.ndarray] = None,
    variance_scale: float = 0.01,
    # variance_minimal: float = 1e30,
    #   variance_replace: Union[None, float] = None
) -> Union[np.ndarray, float]:
    """
    Create a variance field based on the input data and specified parameters.
    Parameters:
    -----------
    y : np.ndarray
        The input data array for which the variance field is to be created.
    variance : Union[bool, np.ndarray], optional
        If True, the variance is calculated as the scaled absolute value of `y`.
        If False, the variance is set to 1.
        If an array, it is used directly as the variance.
        Default is True.
    variance_scale : float, optional
        The scaling factor applied to the absolute value of `y` to calculate the variance.
        Default is 0.01.

    Returns:
    --------
    np.ndarray
        The calculated variance field based on the input data and specified parameters.
    """
    # variance_minimal : float, optional
    #     The minimal threshold for the variance. Values below this threshold are replaced.
    #     Default is 1e-12.
    # variance_replace : Union[None, float], optional
    #     The value to replace variances below the minimal threshold. If None, the minimum
    #     non-NaN variance is used. Default is None.

    # also devide by the variance of the data
    if variance == None:
        # we scale the variance by the absolute value of the data
        var = np.abs(variance_scale * (y / np.nanstd(y)))

        # we handle the case where the data is zero, by setting the variance there to the maximum value
        var_nozero = np.where(y != 0, var, np.nan)
        var_truemin = np.nanmin(var_nozero)

        # we replace the zero values with the minimum
        var = np.where(var != 0, var, var_truemin)

        # we replace the variances below the minimal threshold
        # var = np.where(var <= variance_minimal, var, var_min)

    elif isinstance(variance, (np.ndarray, int, float)):
        var = variance
    else:
        raise TypeError(
            f"The variance parameter must be either None, a float or a numpy array.\nBut is of type: {type(variance)}."
        )
    return var

    # Ln Normal distribution and corresponding cost function


def __annotation_dict__(func: Callable):
    """This funciton returns a TypedDict from a function"""

    d = dict(signature(func).parameters)

    annotations = dict()
    # defaults = dict()

    for key in d:
        annotation, default = d[key].annotation, d[key].default
        annotations[key] = annotation
        # defaults[key] = default

    return annotations


class LeastSquareFit:
    """
    A class to perform least squares fitting using scipy.optimize's least_squares function.
    The fit works also for multidimensional data.
    We try to estiamte the parameters (x) of the funciton F(t, x). With x can be a vector of parameters.
    The dependent variable in this case is y, which is a function of t and x: y = F(t, x)
    The provided function needs to be in the form of y = F(t, x), where t is the independent variable and x are the parameters to be estimated.
    The cost function is used to minimize the difference between the predicted and the actual data.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:
        fit(repetitions: int = 1):
            Perform the fitting process. Can repeat the fitting multiple times.

        predict(t_test: Union[np.ndarray, xr.DataArray]) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
            Predict the dependent variable using the fitted model for given test data.
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: Union[np.ndarray, xr.DataArray],
        y_train: Union[np.ndarray, xr.DataArray],
        cost_func: Union[Callable, None] = None,
        func_kwargs: dict = dict(),
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the LeastSquareFit instance.

        Parameters:
            name (str): The name of the fitting instance.
            func (Callable): The model function to fit.
            cost_func (Callable or None): The cost function to minimize. If None, a default cost function is used based on the model function.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
            y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
            fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
            plot_kwargs (Dict): Additional keyword arguments for plotting.
            parameters (Dict): Final parameters for the model function after the fit.
        """
        self.name = name
        self.func = func

        # set the cost function
        self.cost_func = cost_func
        self.x0 = x0
        self.bounds = bounds

        assert t_train.shape == y_train.shape, "Shape missmatch t_train, y_train"

        self.t_train = t_train
        self.y_train = y_train
        self.func_kwargs = func_kwargs
        self.fit_kwargs = fit_kwargs
        self.plot_kwargs = plot_kwargs
        self.fit_result = None

    def __default_cost_func__(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        The cost function to minimize.

        Parameters:
            x (np.ndarray): The parameters to estimate.
            t (np.ndarray): The independent variable.
            y (np.ndarray): The dependent variable.

        Returns:
            np.ndarray: The difference between the predicted and the actual data.
        """
        diff = y - self.func(t, *x, **kwargs)

        diff = np.ravel(diff)

        # only use the non-NaN values
        idx = np.where(~np.isnan(diff))
        diff = diff[idx]
        return diff

    def __weighted_t_values_cost_func__(
        self, x: np.ndarray, t: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        The cost function to minimize.

        Parameters:
            x (np.ndarray): The parameters to estimate.
            t (np.ndarray): The independent variable.
            y (np.ndarray): The dependent variable.

        Returns:
            np.ndarray: The difference between the predicted and the actual data.
        """
        weight = t**self.t_weight_power
        diff = weight * (y - self.func(t, *x))

        diff = np.ravel(diff)

        # only use the non-NaN values
        idx = np.where(~np.isnan(diff))
        diff = diff[idx]
        return diff

    def __weighted_cost_func__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply a weighted factor to the cost function

        Parameters:
        -----
        """

        y_fit = self.func(t, *x, **kwargs)
        diff = y - y_fit
        diff, weight = np.ravel(diff), np.ravel(self.weight)

        idx = np.where(~np.isnan(diff))
        diff = diff[idx]

        total_loss = np.sum(diff)

        diff = diff * weight

        diff = diff / np.sum(diff) * total_loss

        return diff

    # The model function
    @property
    def func(self):
        """The model function to fit."""
        return self._func

    @func.setter
    def func(self, func: Callable):
        self._func = func

    @func.getter
    def func(self) -> Callable:
        return self._func

    # The cost function to minimize in the least squares fitting
    @property
    def cost_func(self):
        """
        The cost function to minimize.
        If no cost function is provided, a default cost function is used based on the model function.
        It is explained in the __default_cost_func__ method.
        """
        return self._cost_func

    @cost_func.setter
    def cost_func(self, cost_func: Union[Callable, None] = None):
        # set the cost function to the default cost function if None is given
        if cost_func is None:
            self._cost_func = self.__default_cost_func__
        else:
            self._cost_func = cost_func

    @cost_func.getter
    def cost_func(self):
        return self._cost_func

    # ------------
    # Properties for parameters
    @property
    def x0(self):
        """Initial guess for the parameters."""
        return self._x0

    @x0.setter
    def x0(self, x0: np.ndarray):
        # validate that x0 fits the model function
        # self.ParameterDict()
        keys = list(self.ParameterDict.keys())
        # if len(x0) != len(keys) - 1:
        #     raise ValueError(
        #         f"Initial guess x0 has {len(x0)} elements, but the model function has {len(keys)-1} parameters with parameter keys {keys[1:]}"
        #     )

        self._x0 = x0

    @x0.getter
    def x0(self) -> np.ndarray:
        return self._x0

    @property
    def x_guess(self):
        """
        The guess of the parameters.
        It is updated after each fit.
        This property is read-only.
        """

        if self.fit_result is None:
            return self.x0
        else:
            return self.fit_result.x

    def __x_to_parameters__(self, x: np.ndarray) -> dict:
        """This function converts the list of parameter values x to a dictionary with the parameter names given by the function annotations."""
        annotations = __annotation_dict__(self.func)
        keys = list(annotations.keys())
        # ignore the first argument, as it is the independent variable
        keys = keys[1:]

        return dict(zip(keys, x))

    @property
    def parameters(self):
        """
        Final parameters for the model function after the fit.
        The parameters are stored in a dictionary with the parameter names given by the function annotations.
        This property is read-only.
        """
        return self.__x_to_parameters__(self.x_guess)

    @property
    def ParameterDict(self):
        """
        The TypedDict for the parameters of the model function.
        This property is read-only.
        """
        return __annotation_dict__(self.func)

    @property
    def bounds(self):
        """
        Bounds on the parameters.
        This is a Bounds object from scipy.optimize.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: Bounds):
        self._bounds = bounds

    @bounds.getter
    def bounds(self) -> Bounds:
        return self._bounds

    def fit(self, repetitions: int = 1, add_noise: bool = True, seed=42):
        """
        Perform the fitting process. Can repeat the fitting multiple times.

        Parameters:
            repetitions (int): The number of times to repeat the fitting process. Default is 1.

        Returns:
            The result of the fitting process.
        """
        # np.random.seed(seed)

        for i in np.arange(repetitions):

            # x_guess = self.x_guess
            # # add a 5 % of noise to each parameter
            # if add_noise:
            #     x_guess = x_guess + 0.05 * x_guess * np.random.randn(len(x_guess))

            least_squares_kwargs = self.fit_kwargs.copy()
            least_squares_kwargs.update(kwargs=self.func_kwargs)

            self.fit_result = least_squares(
                self.cost_func,
                x0=self.x_guess,
                bounds=self.bounds,
                args=(np.ravel(self.t_train), np.ravel(self.y_train)),
                **least_squares_kwargs,
            )

    def predict(
        self, t_test: Union[np.ndarray, xr.DataArray]
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Predict the dependent variable using the fitted model for given test data.

        Parameters:
            t_test (Union[np.ndarray, xr.DataArray]): Test data for the independent variable.

        Returns:
            Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]: The test data and the predicted dependent variable.
        """

        kwargs = self.parameters.copy()
        kwargs.update(self.func_kwargs)
        return t_test, self.func(t_test, **kwargs)


class DoubleLnNormalFit(LeastSquareFit):
    """
    A class to perform least squares fitting for a double log-normal distribution.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: Union[xr.DataArray, np.ndarray],
        y_train: Union[xr.DataArray, np.ndarray],
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
        t_weight_power: Union[None, int] = None,
    ):
        """
        Initialize the DoubleLnNormalFit instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """

        super().__init__(
            name=name,
            func=double_log_normal_distribution_all,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )

        if t_weight_power is not None:
            self.t_weight_power = t_weight_power
            self.cost_func = self.__weighted_t_values_cost_func__
        else:
            self.cost_func = self.__default_cost_func__


class CleoDoubleLnNormalFit(LeastSquareFit):
    """
    A class to perform least squares fitting for a double log-normal distribution.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: Union[xr.DataArray, np.ndarray],
        y_train: Union[xr.DataArray, np.ndarray],
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
        t_weight_power: Union[None, int] = None,
    ):
        """
        Initialize the DoubleLnNormalFit instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """

        super().__init__(
            name=name,
            func=double_ln_normal_distribution,
            # cost_func=double_ln_normal_distribution_cost,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )

        if t_weight_power is not None:
            self.t_weight_power = t_weight_power
            self.cost_func = self.__weighted_t_values_cost_func__
        else:
            self.cost_func = self.__default_cost_func__


class LnNormalFit(LeastSquareFit):
    """
    A class to perform least squares fitting for a log-normal distribution.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    __func = ln_normal_distribution

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: np.ndarray,
        y_train: np.ndarray,
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the LnNormalFit instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """
        super().__init__(
            name=name,
            func=ln_normal_distribution,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )


class LinearLeastSquare(LeastSquareFit):
    """
    A class to perform least squares fitting for a split linear function.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: np.ndarray,
        y_train: np.ndarray,
        func_kwargs: dict = dict(),
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the SplitLinearLeastSquare instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """
        super().__init__(
            name=name,
            func=linear_func,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            func_kwargs=func_kwargs,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )


class SplitLinearLeastSquare(LeastSquareFit):
    """
    A class to perform least squares fitting for a split linear function.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: np.ndarray,
        y_train: np.ndarray,
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the SplitLinearLeastSquare instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """
        super().__init__(
            name=name,
            func=split_linear_func,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )


class SaturatedLinearLeastSquare(LeastSquareFit):
    """
    A class to perform least squares fitting for a saturated linear function.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: np.ndarray,
        y_train: np.ndarray,
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the SaturatedLinearLeastSquare instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
        """
        super().__init__(
            name=name,
            func=saturated_linear_func,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
        )


class FixedSaturatedLinearLeastSquare(LeastSquareFit):
    """
    A class to perform least squares fitting for a saturated linear function for a fixed saturation value.

    Attributes:
        name (str): The name of the fitting instance.
        func (Callable): The model function to fit.
        cost_func (Callable): The cost function to minimize.
        x0 (np.ndarray): Initial guess for the parameters.
        bounds (Bounds): Bounds on the parameters.
        t_train (Union[np.ndarray, xr.DataArray]): Training data for the independent variable.
        y_train (Union[np.ndarray, xr.DataArray]): Training data for the dependent variable.
        fit_kwargs (Dict): Additional keyword arguments for the least_squares function.
        plot_kwargs (Dict): Additional keyword arguments for plotting.
        fit_result: The result of the fitting process.

    Methods:

    """

    def __init__(
        self,
        name: str,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: np.ndarray,
        y_train: np.ndarray,
        saturation_value: float,
        func_kwargs: dict = dict(),
        fit_kwargs: dict = dict(),
        plot_kwargs: dict = dict(),
        weight: Union[np.ndarray, None] = None,
    ):
        """
        Initialize the SaturatedLinearLeastSquare instance.

        Parameters:
            name (str): The name of the fitting instance.
            x0 (np.ndarray): Initial guess for the parameters.
            bounds (Bounds): Bounds on the parameters.
            t_train (np.ndarray): Training data for the independent variable.
            y_train (np.ndarray): Training data for the dependent variable.
            saturation_values (float): The fixed saturation value for the function.
        """

        self.weight = weight
        self.saturation_value = saturation_value
        func_kwargs.update(saturation_value=saturation_value)

        super().__init__(
            name=name,
            func=saturated_linear_func,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            func_kwargs=func_kwargs,
            plot_kwargs=plot_kwargs,
        )

        # we need to set the saturation value in the kwargs for the cost function

        if self.weight is not None:
            if isinstance(self.weight, np.ndarray) or isinstance(self.weight, xr.DataArray):
                assert self.weight.shape == self.y_train.shape, "Shape missmatch weight, y_train"
                self.cost_func = self.__weighted_cost_func__
            else:
                raise ValueError("The weight parameter must be a numpy array or a xarray DataArray")

    @property
    def full_parameters(self):
        """
        Final parameters for the model function after the fit.
        The parameters are stored in a dictionary with the parameter names given by the function annotations.
        This property is read-only.
        """

        # if no saturation

        parameters = self.parameters
        x_s = (self.saturation_value - parameters["f_0"]) / parameters["slope_1"]

        parameters.update(x_split=x_s)
        parameters.update(slope_2=0.0)

        return parameters


# Classes for handling the LogNormal Parameters in the individual spaces


class LogNormalDistribution:

    @staticmethod
    def linearspace(
        x: Union[float, np.ndarray, xr.DataArray],
        mu: Union[float, np.ndarray, xr.DataArray],
        sigma: Union[float, np.ndarray, xr.DataArray],
        scale: Union[float, np.ndarray, xr.DataArray],
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute L(x) for the given x."""
        return (
            scale
            * 1
            / (x * sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * (np.log(x) - mu) ** 2 / sigma**2)
        )

    @staticmethod
    def logspace(
        y: Union[float, np.ndarray, xr.DataArray],
        mu: Union[float, np.ndarray, xr.DataArray],
        sigma: Union[float, np.ndarray, xr.DataArray],
        scale: Union[float, np.ndarray, xr.DataArray],
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute l(y) for the given y."""
        return scale * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(y) - mu) ** 2 / sigma**2)


class StandardizedParameters:
    """
    Represents the full set of log-normal parameters derived from the base triple.

    Parameters:
        mu_L: Union[float, np.ndarray, xr.DataArray]
            Mean of the log-normal distribution in log space (L(x)).
        sigma_L: Union[float, np.ndarray, xr.DataArray]
            Standard deviation of the log-normal distribution in log space (L(x)).
        scale_L: Union[float, np.ndarray, xr.DataArray]
            Scale factor for the L(x) distribution.
    """

    def __init__(
        self,
        mu_L: Union[float, np.ndarray, xr.DataArray],
        sigma_L: Union[float, np.ndarray, xr.DataArray],
        scale_L: Union[float, np.ndarray, xr.DataArray],
    ):
        self.mu_L = mu_L
        self.sigma_L = sigma_L
        self.scale_L = scale_L

    # make the individual parameters accessible by dict like access
    def __getitem__(self, key: str) -> Union[float, np.ndarray, xr.DataArray]:
        return self.__getattribute__(key)

    @property
    def mu_l(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute mu_l for l(y)."""
        return self.mu_L - self.sigma_L**2

    @property
    def sigma(self) -> Union[float, np.ndarray, xr.DataArray]:
        return self.sigma_L

    @property
    def sigma_l(self) -> Union[float, np.ndarray, xr.DataArray]:
        return self.sigma_L

    @property
    def x_max(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute x_max (mode of L(x))."""
        return np.exp(self.mu_l)

    @property
    def maximum_value(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute maximum value of L(x)."""
        factor = LogNormalDistribution.linearspace(
            x=self.x_max,
            mu=self.mu_L,
            sigma=self.sigma_L,
            scale=1,
        )
        return self.scale_L * factor

    @property
    def scale_l(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute scale_l for l(y)."""
        factor = LogNormalDistribution.logspace(
            y=self.x_max,
            mu=self.mu_l,
            sigma=self.sigma_l,
            scale=1,
        )
        return self.maximum_value / factor

    @property
    def geometric_mu_L(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute geometric mean of the distribution."""
        return np.exp(self.mu_L)

    @property
    def geometric_mu_l(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute geometric mean of the distribution."""
        return np.exp(self.mu_l)

    @property
    def geometric_std_dev(self) -> Union[float, np.ndarray, xr.DataArray]:
        """Compute geometric standard deviation of the distribution."""
        return np.exp(self.sigma)

    def __params_to_dict__(
        self,
        get_keys: Tuple[str, str, str],
        dict_keys: Tuple[str, str, str],
    ) -> dict:

        return {dict_keys[i]: getattr(self, get_keys[i]) for i in range(3)}

    def get_parameters_linear(self, dict_keys: Union[None, Tuple[str, str, str]] = None) -> dict:
        get_keys = ("mu_L", "sigma_L", "scale_L")
        if dict_keys is None:
            dict_keys = get_keys

        return self.__params_to_dict__(get_keys, dict_keys)

    def get_parameters_log(self, dict_keys: Union[None, Tuple[str, str, str]] = None) -> dict:
        get_keys = ("mu_l", "sigma_l", "scale_l")
        if dict_keys is None:
            dict_keys = get_keys

        return self.__params_to_dict__(get_keys, dict_keys)

    def get_geometric_parameters_linear(
        self, dict_keys: Union[None, Tuple[str, str, str]] = None
    ) -> dict:
        get_keys = ("geometric_mu_L", "geometric_std_dev", "scale_L")
        if dict_keys is None:
            dict_keys = get_keys
        return self.__params_to_dict__(get_keys, dict_keys)

    def get_geometric_parameters_log(self, dict_keys: Union[None, Tuple[str, str, str]] = None) -> dict:
        get_keys = ("geometric_mu_l", "geometric_std_dev", "scale_l")
        if dict_keys is None:
            dict_keys = get_keys
        return self.__params_to_dict__(get_keys, dict_keys)

    def get_maximum_values(self, dict_keys: Union[None, Tuple[str, str, str]] = None) -> dict:
        get_keys = ("x_max", "sigma", "maximum_value")
        if dict_keys is None:
            dict_keys = get_keys
        return self.__params_to_dict__(get_keys, dict_keys)

    def summary(self) -> dict:
        """Provide a summary of all parameters."""
        return {
            "Linear": {
                "mu_L": self.mu_L,
                "sigma_L": self.sigma_L,
                "scale_L": self.scale_L,
            },
            "Log": {
                "mu_l": self.mu_l,
                "sigma_l": self.sigma_l,
                "scale_l": self.scale_l,
            },
            "x": {
                "x_max": self.x_max,
                "y_max": self.maximum_value,
                "sigma": self.sigma_L,
            },
            "geometric": {
                "geometric_mu_L": self.geometric_mu_L,
                "geometric_std_dev_L": self.geometric_std_dev,
            },
        }

    def __repr__(self):
        res = f"StandardizedParameters"
        summary = self.summary()
        for key in summary.keys():
            res += f"\n{key}:"
            for k, v in summary[key].items():
                res += f"\n\t{k}: {v}"
            res += "\n-----------------"

        return res


class LogNormalParameters:
    """Abstract base class for parameter sets."""

    def to_base_triple(self):
        """Convert to the canonical form: mu_L, sigma_L, scale_L."""
        pass

    def standardize(self):
        """Convert the Child Class to the standard class."""
        return StandardizedParameters(*self.to_base_triple())

    # make the individual parameters accessible by dict like access
    def __getitem__(self, key: str) -> Union[float, np.ndarray, xr.DataArray]:
        return self.__dict__()[key]


class MaximumPointGeometricSigma(LogNormalParameters):
    """
    Represents the log-normal distribution using the mode (x_max)
    and maximum value (max_l) of l(x), and sigma_l.

    Parameters:
        x_max: Union[float, np.ndarray, xr.DataArray]
            The mode of the distribution (x_max).
        geometric_sigma_L: Union[float, np.ndarray, xr.DataArray]
            The geometric standard deviation of the distribution in log space.
        max_l: Union[float, np.ndarray, xr.DataArray]
            The maximum value of L(x).
    """

    def __init__(
        self,
        x_max: Union[float, np.ndarray, xr.DataArray],
        geometric_std_dev_l: Union[float, np.ndarray, xr.DataArray],
        maximum_value: Union[float, np.ndarray, xr.DataArray],
    ):
        self.x_max = x_max
        self.geometric_std_dev_l = geometric_std_dev_l
        self.maximum_value = maximum_value

    def to_base_triple(
        self,
    ) -> Tuple[
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
    ]:
        """Convert to the base triple (mu_L, sigma_L, scale_L)."""
        sigma_L = np.log(self.geometric_std_dev_l)
        mu_L = np.log(self.x_max) + sigma_L**2
        x_max = self.x_max
        factor = (
            1
            / (np.sqrt(2 * np.pi) * sigma_L * x_max)
            * np.exp(-0.5 * (np.log(x_max) - mu_L) ** 2 / sigma_L**2)
        )
        scale_L = self.maximum_value / factor
        return mu_L, sigma_L, scale_L


class MuSigmaScaleLinear(LogNormalParameters):
    """
    Represents the log-normal distribution using mu_L, sigma_L, and scale_L
    directly, and transforms them into their respective child classes.

    Parameters:
        mu_L: Union[float, np.ndarray, xr.DataArray]
            Mean of the log-normal distribution in log space (L(x)).
        sigma_L: Union[float, np.ndarray, xr.DataArray]
            Standard deviation of the log-normal distribution in log space (L(x)).
        scale_L: Union[float, np.ndarray, xr.DataArray]
            Scale factor for the L(x) distribution.
    """

    def __init__(
        self,
        mu_L: Union[float, np.ndarray, xr.DataArray],
        sigma_L: Union[float, np.ndarray, xr.DataArray],
        scale_L: Union[float, np.ndarray, xr.DataArray],
    ):
        self.mu_L = mu_L
        self.sigma_L = sigma_L
        self.scale_L = scale_L

    def to_base_triple(
        self,
    ) -> Tuple[
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
    ]:
        """Convert to the base triple (mu_L, sigma_L, scale_L)."""
        return self.mu_L, self.sigma_L, self.scale_L


class MuSigmaScaleLog(LogNormalParameters):
    """
    Represents the log-normal distribution using mu_l, sigma_l, and scale_l
    directly, and transforms them into their respective child classes.

    Parameters:
        mu_l: Union[float, np.ndarray, xr.DataArray]
            Mean of the log-normal distribution in log space (l(x)).
        sigma_l: Union[float, np.ndarray, xr.DataArray]
            Standard deviation of the log-normal distribution in log space (l(x)).
        scale_l: Union[float, np.ndarray, xr.DataArray]
            Scale factor for the l(x) distribution.
    """

    def __init__(
        self,
        mu_l: Union[float, np.ndarray, xr.DataArray],
        sigma_l: Union[float, np.ndarray, xr.DataArray],
        scale_l: Union[float, np.ndarray, xr.DataArray],
    ):
        self.mu_l = mu_l
        self.sigma_l = sigma_l
        self.scale_l = scale_l

    def to_base_triple(
        self,
    ) -> Tuple[
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
    ]:
        """Convert to the base triple (mu_L, sigma_L, scale_L)."""
        mu_L = self.mu_l + self.sigma_l**2
        x_max = np.exp(self.mu_l)
        sigma_L = self.sigma_l
        factor_l = LogNormalDistribution.logspace(
            y=x_max,
            mu=self.mu_l,
            sigma=self.sigma_l,
            scale=1,
        )
        factor_L = LogNormalDistribution.linearspace(
            x=x_max,
            mu=mu_L,
            sigma=sigma_L,
            scale=1,
        )

        scale_L = self.scale_l * factor_l / factor_L
        return mu_L, self.sigma_l, scale_L


class GeometricMuSigmaScaleLog(LogNormalParameters):
    """
    Represents the log-normal distribution using mu_l, sigma_l, and scale_l
    directly, and transforms them into their respective child classes.

    Parameters:
        mu_l: Union[float, np.ndarray, xr.DataArray]
            Mean of the log-normal distribution in log space (l(x)).
        sigma_l: Union[float, np.ndarray, xr.DataArray]
            Standard deviation of the log-normal distribution in log space (l(x)).
        scale_l: Union[float, np.ndarray, xr.DataArray]
            Scale factor for the l(x) distribution.
    """

    def __init__(
        self,
        geometric_mu_l: Union[float, np.ndarray, xr.DataArray],
        geometric_std_dev: Union[float, np.ndarray, xr.DataArray],
        scale_l: Union[float, np.ndarray, xr.DataArray],
    ):
        self.geometric_mu_l = geometric_mu_l
        self.geometric_std_dev = geometric_std_dev
        self.scale_l = scale_l

        self.mu_l = np.log(self.geometric_mu_l)
        self.sigma_l = np.log(self.geometric_std_dev)

    def to_base_triple(
        self,
    ) -> Tuple[
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
        Union[float, np.ndarray, xr.DataArray],
    ]:
        """Convert to the base triple (mu_L, sigma_L, scale_L)."""
        mu_L = self.mu_l + self.sigma_l**2
        x_max = np.exp(self.mu_l)
        sigma_L = self.sigma_l
        factor_l = LogNormalDistribution.logspace(
            y=x_max,
            mu=self.mu_l,
            sigma=self.sigma_l,
            scale=1,
        )
        factor_L = LogNormalDistribution.linearspace(
            x=x_max,
            mu=mu_L,
            sigma=sigma_L,
            scale=1,
        )

        scale_L = self.scale_l * factor_l / factor_L
        return mu_L, self.sigma_l, scale_L
