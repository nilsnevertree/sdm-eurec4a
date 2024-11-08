from typing import Tuple, Dict, Union, Callable, TypedDict
import numpy as np
import xarray as xr
from scipy.optimize import least_squares, Bounds


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


def split_linear_cost_func(
    x: Tuple[float, float, float, float],
    t: np.ndarray,
    y: np.ndarray,
    variance: Union[None, float, int, np.ndarray] = None,
    variance_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute the cost for the split linear fit.

    Parameters:
        x (Tuple[float, float, float, float]): Parameters for the split_linear_func.
        t (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        variance (Union[None, float, int, np.ndarray], optional): Variance of the dependent variable. Default is None.
        variance_scale (float, optional): Scale factor for the variance. Default is 0.01.

    Returns:
        np.ndarray: The computed cost.
    """

    y_pred = smodels.split_linear_func(t, *x)

    return np.ravel((y_pred - y))


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


# ++++++++++++++++++++++++++++


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


def ln_normal_distribution_cost(
    x: Tuple[float, float, float],
    t: np.ndarray,
    y: np.ndarray,
    variance: Union[None, np.ndarray, float] = None,
    variance_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute the cost for the log-normal distribution fit.

    Parameters:
        x (Tuple[float, float, float]): Parameters for the log-normal distribution.
        t (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        variance (Union[None, np.ndarray, float], optional): Variance of the dependent variable. Default is None.
        variance_scale (float, optional): Scale factor for the variance. Default is 0.01.

    Returns:
        np.ndarray: The computed cost.
    """
    y_pred = ln_normal_distribution(t, *x)

    var = create_variance_field(y, variance, variance_scale)

    return np.ravel((y_pred - y) / np.sqrt(var))


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


def double_ln_normal_distribution(
    t: np.ndarray,
    mu1: float,
    sigma1: float,
    scale_factor1: float,
    mu2: float,
    sigma2: float,
    scale_factor2: float,
) -> np.ndarray:
    """
    Compute the double log-normal distribution.

    Parameters:
        t (np.ndarray): Independent variable.
        mu1 (float): Mean of the first log-normal distribution.
        sigma1 (float): Standard deviation of the first log-normal distribution.
        scale_factor1 (float): Scale factor for the first log-normal distribution.
        mu2 (float): Mean of the second log-normal distribution.
        sigma2 (float): Standard deviation of the second log-normal distribution.
        scale_factor2 (float): Scale factor for the second log-normal distribution.

    Returns:
        np.ndarray: The computed double log-normal distribution.
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

        result += dn_dlnr

    return result


def double_ln_normal_distribution_cost(
    x: Tuple[float, float, float, float, float, float],
    t: np.ndarray,
    y: np.ndarray,
    variance: Union[None, float, int, np.ndarray] = None,
    variance_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute the cost for the double log-normal distribution fit.

    Parameters:
        x (Tuple[float, float, float, float, float, float]): Parameters for the double log-normal distribution.
        t (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        variance (Union[None, float, int, np.ndarray], optional): Variance of the dependent variable. Default is None.
        variance_scale (float, optional): Scale factor for the variance. Default is 0.01.

    Returns:
        np.ndarray: The computed cost.
    """
    y_pred = double_ln_normal_distribution(t, *x)

    var = create_variance_field(y, variance, variance_scale)

    return np.ravel((y_pred - y) / np.sqrt(var))


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
    x: np.ndarray, f_0: float = 2, slope_1: float = 1, saturation_value: float = 1
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


def saturated_linear_cost_func(
    x: Tuple[float, float, float, float],
    t: np.ndarray,
    y: np.ndarray,
    variance: Union[None, float, int, np.ndarray] = None,
    variance_scale: float = 0.01,
) -> np.ndarray:
    """
    Compute the cost for the split linear fit.

    Parameters:
        x (Tuple[float, float, float, float, float]): Parameters for the split_linear_func.
        t (np.ndarray): Independent variable.
        y (np.ndarray): Dependent variable.
        variance (Union[None, float, int, np.ndarray], optional): Variance of the dependent variable. Default is None.
        variance_scale (float, optional): Scale factor for the variance. Default is 0.01.

    Returns:
        np.ndarray: The computed cost.
    """

    y_pred = saturated_linear_func(t, *x)

    return np.ravel((y_pred - y))


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
        cost_func: Callable,
        x0: np.ndarray,
        bounds: Bounds,
        t_train: Union[np.ndarray, xr.DataArray],
        y_train: Union[np.ndarray, xr.DataArray],
        fit_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
    ):
        """
        Initialize the LeastSquareFit instance.

        Parameters:
            name (str): The name of the fitting instance.
            func (Callable): The model function to fit.
            cost_func (Callable): The cost function to minimize.
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
        self.cost_func = cost_func
        self.x0 = x0
        self.bounds = bounds
        self.t_train = t_train
        self.y_train = y_train
        self.fit_kwargs = fit_kwargs
        self.plot_kwargs = plot_kwargs
        self.fit_result = None

    def fit(self, repetitions: int = 1):
        """
        Perform the fitting process. Can repeat the fitting multiple times.

        Parameters:
            repetitions (int): The number of times to repeat the fitting process. Default is 1.

        Returns:
            The result of the fitting process.
        """
        for i in range(repetitions):
            if i != 0:
                x0 = self.fit_result.x  # type: ignore
            else:
                x0 = self.x0

            self.fit_result = least_squares(
                self.cost_func,
                x0=x0,
                bounds=self.bounds,
                args=(np.ravel(self.t_train), np.ravel(self.y_train)),
                **self.fit_kwargs,
            )

        return self.fit_result

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
        self.t_test = t_test
        self.y_test = self.func(self.t_test, *self.fit_result.x)  # type: ignore

        return self.t_test, self.y_test


class DoubleLnNormalLeastSquare(LeastSquareFit):
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

    _func = double_ln_normal_distribution
    _cost_func = double_ln_normal_distribution_cost

    def __init__(
        self, name: str, x0: np.ndarray, bounds: Bounds, t_train: np.ndarray, y_train: np.ndarray
    ):
        """
        Initialize the DoubleLnNormalLeastSquare instance.

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
            cost_func=double_ln_normal_distribution_cost,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
        )


class LnNormalLeastSquare(LeastSquareFit):
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

    _func = ln_normal_distribution
    _cost_func = ln_normal_distribution_cost

    def __init__(
        self, name: str, x0: np.ndarray, bounds: Bounds, t_train: np.ndarray, y_train: np.ndarray
    ):
        """
        Initialize the LnNormalLeastSquare instance.

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
            cost_func=ln_normal_distribution_cost,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
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

    _func = split_linear_func
    _cost_func = ln_normal_distribution_cost

    def __init__(
        self, name: str, x0: np.ndarray, bounds: Bounds, t_train: np.ndarray, y_train: np.ndarray
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
            cost_func=ln_normal_distribution_cost,
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
        )


from typing import Tuple, Union
