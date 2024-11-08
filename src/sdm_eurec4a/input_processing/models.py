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
        self.t_train = t_train
        self.y_train = y_train
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
        diff = self.y_train - self.func(self.t_train, *x)

        return np.ravel(diff)

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
    def cost_func(self, cost_func: Union[Callable, None]):
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
        if len(x0) != len(keys) - 1:
            raise ValueError(
                f"Initial guess x0 has {len(x0)} elements, but the model function has {len(keys)-1} parameters with parameter keys {keys[1:]}"
            )

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

    def fit(self, repetitions: int = 1):
        """
        Perform the fitting process. Can repeat the fitting multiple times.

        Parameters:
            repetitions (int): The number of times to repeat the fitting process. Default is 1.

        Returns:
            The result of the fitting process.
        """
        for i in np.arange(repetitions):

            self.fit_result = least_squares(
                self.cost_func,
                x0=self.x_guess,
                bounds=self.bounds,
                args=(np.ravel(self.t_train), np.ravel(self.y_train)),
                **self.fit_kwargs,
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
        return t_test, self.func(t_test, **self.parameters)


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
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
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
            x0=x0,
            bounds=bounds,
            t_train=t_train,
            y_train=y_train,
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
