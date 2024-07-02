"""Module to transfer the fitted observations to CLEO."""
from __future__ import annotations

import warnings

from typing import Dict, Tuple, Union

import lmfit
import numpy as np
import xarray as xr
import yaml

from scipy.optimize import curve_fit
from sdm_eurec4a.input_processing.models import (
    linear_func,
    lnnormaldist,
    split_linear_func,
)
from sdm_eurec4a.reductions import shape_dim_as_dataarray


class Input:
    """Class to handle the particle size distribution input."""

    def __init__(
        self,
        type: str = "None",
        func: function = lambda x: None,
        independent_vars: list = [],
        parameters: dict = {},
        model_arguments: dict = {},
    ) -> None:
        """Initialize the ParticleSizeDistributionInput class."""
        self.independent_vars = independent_vars
        self.set_parameters(parameters=parameters)
        self.set_type(type=type)
        self.set_func(func=func)
        self.set_model(**model_arguments)
        self.model_result = None

    def set_parameters(self, parameters: dict = dict()) -> None:
        """
        Set the parameters of the particle size distribution.

        Parameters
        ----------
        parameters : dict
            The parameters of the particle size distribution.
            Default is an empty dictionary.

        Returns
        -------
        None
        """
        self.parameters = parameters
        self.__autoupdate_parameters__()

    def get_parameters(self) -> dict:
        """
        Get the parameters of the particle size distribution.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            The parameters of the particle size distribution.
        """
        return self.parameters

    def add_parameters(self, parameters: Dict) -> None:
        """
        Add the model parameters to the parameters of the particle size
        distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        assert (
            parameters.keys() == self.model_parameters.keys()
        ), "The keys of the parameters to add must be the same as the keys of the model parameters."

        for key in parameters.keys():
            value = parameters[key]
            # for parameters in self.parameters which are numpy arrays
            if isinstance(self.parameters[key], np.ndarray):
                # convert the parameter input to an numpy array
                if isinstance(value, (float, int)):
                    value = np.array([value])
                elif isinstance(value, list):
                    value = np.array(value)
                elif isinstance(value, (np.ndarray,)):
                    pass
                else:
                    raise TypeError(
                        f"The type of the parameter {key} from the input parameters is not supported."
                    )

                self.parameters[key] = np.concatenate((self.parameters[key], value))

            elif isinstance(self.parameters[key], list):
                self.parameters[key].append(parameters[key])
            else:
                raise TypeError(
                    f"The type of the parameter {key} from self.parameters is not supported. Only np.ndarray and list are supported."
                )
        self.__autoupdate_parameters__()

    def __autoupdate_parameters__(self) -> None:
        """
        Autoupdate all dependent parameters. The default method changes
        nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def set_type(self, type: str = "Not specified") -> None:
        """
        Set the type of the particle size distribution.

        Parameters
        ----------
        type : str
            The type of the particle size distribution.
            Default is "Not specified".

        Returns
        -------
        None
        """
        self.type: str = type

    def get_type(self) -> str:
        """
        Get the type of the particle size distribution.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The type of the particle size distribution.
        """
        return self.type

    def set_func(self, func: function = lambda x: None) -> None:
        """
        Set the function of the particle size distribution.

        Parameters
        ----------
        func : function
            The function of the particle size distribution.
            Default is a function that returns None.

        Returns
        -------
        None
        """
        self.func: function = func

    def get_func(self) -> function:
        """
        Get the function of the particle size distribution.

        Parameters
        ----------
        None

        Returns
        -------
        function
            The function of the particle size distribution.
        """
        return self.func

    def set_model(self, **kwargs: dict()) -> None:
        """
        Set the model of the particle size distribution. It uses the
        lmfit.model.Model class. The function to be used for this is the
        function set in the set_func method. The function cannot be provided as
        a parameter to this method! It is set in the set_func method.

        The Model can be used to fit the particle size distribution to the data.


        Parameters
        ----------
        **kwargs : dict
            The keyword arguments for the lmfit.model.Model class.
            See also the lmfit.model.Model class.
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model

        Returns
        -------
        None
        """
        self.model: lmfit.model.Model = lmfit.model.Model(
            func=self.get_func(), independent_vars=self.independent_vars, **kwargs
        )
        # make sure that radii is the independent variable
        self.__set_model_independ_vars__()
        self.__update_model__()
        self.set_model_parameters(self.model.make_params())
        self.update_model_parameters()

    def __set_model_independ_vars__(self) -> None:
        """
        Set the independent variables of the particle size distribution.

        Parameters
        ----------
        independent_vars : list
            The independent variables of the particle size distribution.

        Returns
        -------
        None
        """
        self.model.independent_vars = self.independent_vars

    def __update_model__(self) -> None:
        """
        Update the model of the particle size distribution. The default method
        changes nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def get_model(self) -> lmfit.model.Model:
        """
        Get the model of the particle size distribution.

        Parameters
        ----------
        None

        Returns
        -------
        lmfit.model.Model
            The model of the particle size distribution.
        """
        return self.model

    def set_model_parameters(self, model_parameters: lmfit.Parameters) -> None:
        """
        Set the model parameters of the particle size distribution.

        Parameters
        ----------
        model_parameters : lmfit.Parameters
            The model parameters of the particle size distribution.

        Returns
        -------
        None
        """
        self.model_parameters = model_parameters

    def get_model_parameters(self) -> lmfit.Parameters:
        return self.model_parameters

    def update_model_parameters(self) -> None:
        """
        Update the model parameters of the particle size distribution. The
        default method changes nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def update_individual_model_parameters(self, parameter: lmfit.Parameter) -> None:
        """
        Update a specific model parameter.

        Parameters
        ----------
        parameter : lmfit.Parameters
            The model parameters of the particle size distribution.

        Returns
        -------
        None
        """
        name = parameter.name
        self.model_parameters[name] = parameter

    def lmfitParameterValues_to_dict(self, parameters: lmfit.Parameters, add: bool = True) -> dict:
        result = dict()
        for key in parameters:
            result[key] = parameters[key].value

        if add is True:
            self.add_parameters(parameters=result)
        return result

    def lmfitParameterStderr_to_dict(self, parameters: lmfit.Parameters) -> dict:
        result = dict()
        for key in parameters:
            result[key] = parameters[key].stderr

        return result


class ThermodynamicLinear(Input):
    """Class to handle the thermodynamic input."""

    def __init__(
        self,
        f_0: np.ndarray = np.empty(0),
        slope: np.ndarray = np.empty(0),
    ) -> None:
        """
        Initialize the PSD_LnNormal class.

        This class is a subclass of the ParticleSizeDistributionInput class.
        It is used to handle the particle size distribution input of the type "LnNormal".

        Parameters
        ----------
        f_0 : float
            The y-intercept.
        slope : float
            The slope of the linear function.


        Returns
        -------
        None
        """

        params = dict(
            f_0=f_0,
            slope=slope,
        )
        super().__init__(
            type="Linear",
            func=linear_func,
            independent_vars=["x"],
            parameters=params,
        )

        self.__autoupdate_parameters__()

    def get_f_0(self):
        return self.get_parameters()["f_0"]

    def get_slope(self):
        return self.get_parameters()["slope"]

    def __str__(self):
        f_0 = f"intercept = {self.get_f_0()}"
        slope = f"slope = {self.get_slope()}"
        return "\n".join([f_0, slope])

    def eval_func(self, x: np.ndarray) -> Tuple[np.ndarray]:
        """
        Evaluate the model of the particle size distribution.

        Parameters
        ----------
        **kwargs : dict
            The keyword arguments for the model function.
            See also the lmfit.model.Model.eval.
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.eval

        Returns
        -------
        np.ndarray
            The evaluated model of the particle size distribution.
        """
        params = self.get_parameters()

        result = self.func(
            x=x,
            f_0=params["f_0"][0],
            slope=params["slope"][0],
        )
        # For each mode, evaluate the model

        return np.array(result)


class ThermodynamicSplitLinear(Input):
    """Class to handle the thermodynamic input."""

    def __init__(
        self,
        x_split: np.ndarray = np.empty(0),
        f_0: np.ndarray = np.empty(0),
        slope_1: np.ndarray = np.empty(0),
        slope_2: np.ndarray = np.empty(0),
    ) -> None:
        """
        Initialize the PSD_LnNormal class.

        This class is a subclass of the ParticleSizeDistributionInput class.
        It is used to handle the particle size distribution input of the type "LnNormal".

        Parameters
        ----------
        geometric_means : np.ndarray
            The geometric means.
            Default is an empty array.
        geometric_sigmas : np.ndarray
            The geometric sigma or geometric standard deviation.
            Default is an empty array.
        scale_factors : np.ndarray
            The scale factors.
            Default is an empty array.

        Returns
        -------
        None
        """
        params = dict(
            f_0=f_0,
            slope_1=slope_1,
            slope_2=slope_2,
            x_split=x_split,
        )
        super().__init__(
            type="SplitLinear",
            func=split_linear_func,
            independent_vars=["x"],
            parameters=params,
        )

        self.__autoupdate_parameters__()

    def __autoupdate_parameters__(self) -> None:
        self.set_mode_number()

        slopes = list()
        slope_1 = self.get_parameters()["slope_1"]
        slope_2 = self.get_parameters()["slope_2"]
        for n in range(self.get_mode_number()):
            slopes.append((slope_1[n], slope_2[n]))
        slopes = np.asarray(slopes)
        self.set_slopes(slopes)

    def set_slopes(self, slopes):
        self.slopes = slopes

    def get_slopes(self) -> list:
        return self.slopes

    def set_mode_number(self) -> None:
        """
        Update the mode number based on the length of the geometric means.

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        params = self.get_parameters()
        nmodes = len(params["f_0"])
        self.number_modes = nmodes

    def get_mode_number(self) -> int:
        """
        Get the mode number.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The mode number.
        """
        return self.number_modes

    def get_f_0(self):
        return self.get_parameters()["f_0"]

    def get_x_split(self):
        return self.get_parameters()["x_split"]

    def __str__(self):
        nmodes = f"nmodes = {self.get_mode_number():.2e}"

        f_0s = ""
        x_split = ""
        slopes = ""

        if self.get_mode_number() <= 0:
            return "No modes found"

        else:
            for i in range(self.get_mode_number()):
                f_0s += f"{self.get_f_0()[i]:.2e}, "
                x_split += f"{self.get_x_split()[i]:.2e}, "
                slopes += f"{self.get_slopes()[i]}, "

        f_0s = f"intercepts = [{f_0s}]"
        x_split = f"split x value = [{x_split}]"
        slopes = f"slopes = [{slopes}]"
        return "\n".join([nmodes, f_0s, x_split, slopes])

    def eval_func(self, x: np.ndarray) -> Tuple[np.ndarray]:
        """
        Evaluate the model of the particle size distribution.

        Parameters
        ----------
        **kwargs : dict
            The keyword arguments for the model function.
            See also the lmfit.model.Model.eval.
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.eval

        Returns
        -------
        np.ndarray
            The evaluated model of the particle size distribution.
        """
        params = self.get_parameters()

        result = list()

        for idx in range(self.get_mode_number()):
            result.append(
                self.func(
                    x=x,
                    f_0=params["f_0"][idx],
                    slope_1=params["slope_1"][idx],
                    slope_2=params["slope_2"][idx],
                    x_split=params["x_split"][idx],
                )
            )
        # For each mode, evaluate the model

        return np.array(result)


class PSD_LnNormal(Input):
    """
    A class used to represent the particle size distribution input using a
    lognormal distribution.

    This class is a subclass of the ParticleSizeDistributionInput class and is used to handle the particle size distribution input of the type "LnNormal". It provides methods to set and get geometric means, geometric sigmas, scale factors, mode number, and number concentration. It also provides methods to update model parameters and auto-update all dependent parameters.

    Attributes
    ----------
    number_modes : int
        The number of modes in the distribution.
    parameters : dict
        A dictionary to store geometric means, geometric sigmas, scale factors, and number concentration.
    model_parameters : lmfit.Parameters
        The model parameters of the particle size distribution.
    model : lmfit.model.Model
        The model used for fitting the particle size distribution. Initialized in the constructor.
    model_result : lmfit.model.ModelResult
        The result of the model fitting.
        Initialized as None in the constructor and updated after fitting.

    Methods
    -------
    set_mode_number():
        Updates the mode number based on the length of the geometric means.
    get_mode_number():
        Returns the mode number.
    set_geometric_means(geometric_means):
        Sets the geometric means.
    get_geometric_means():
        Returns the geometric means.
    set_geometric_sigmas(geometric_sigmas):
        Sets the geometric sigmas.
    get_geometric_sigmas():
        Returns the geometric sigmas.
    set_scale_factors(scale_factor):
        Sets the scale factors.
    get_scale_factors():
        Returns the scale factors.
    set_number_concentration():
        Sets the number concentration.
    get_number_concentration():
        Returns the number concentration.
    update_model_parameters(params):
        Updates the model parameters of the particle size distribution.
    update_individual_model_parameters(parameter):
        Updates a specific model parameter.
    __autoupdate_parameters__():
        Auto-updates all dependent parameters.
    __add__(other):
        Returns a new PSD_LnNormal instance that is the sum of this instance and another.
    __str__():
        Returns a string representation of the PSD_LnNormal instance.
    """

    def __init__(
        self,
        geometric_means: np.ndarray = np.empty(0),
        geometric_sigmas: np.ndarray = np.empty(0),
        scale_factors: np.ndarray = np.empty(0),
    ) -> None:
        """
        Initialize the PSD_LnNormal class.

        This class is a subclass of the ParticleSizeDistributionInput class.
        It is used to handle the particle size distribution input of the type "LnNormal".

        Parameters
        ----------
        geometric_means : np.ndarray
            The geometric means.
            Default is an empty array.
        geometric_sigmas : np.ndarray
            The geometric sigma or geometric standard deviation.
            Default is an empty array.
        scale_factors : np.ndarray
            The scale factors.
            Default is an empty array.

        Returns
        -------
        None
        """
        super().__init__(
            type="LnNormal",
            func=lnnormaldist,
            independent_vars=["radii"],
            parameters=dict(),
        )

        if len(geometric_means) != len(geometric_sigmas) != len(scale_factors):
            message = "The length of the geometric means, geometric sigmas, and scale factors must be the same."
            message += (
                f"\nBut is {len(geometric_means)}, {len(geometric_sigmas)}, and {len(scale_factors)}."
            )
            raise ValueError(message)
        else:
            self.set_geometric_means(geometric_means)
            self.set_geometric_sigmas(geometric_sigmas)
            self.set_scale_factors(scale_factors)
            # The automatic update does the same as:
            # self.set_mode_number()
            # self.set_number_concentration()
            self.__autoupdate_parameters__()

    def set_mode_number(self) -> None:
        """
        Update the mode number based on the length of the geometric means.

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        nmodes = len(self.get_geometric_means())
        self.number_modes = nmodes

    def get_mode_number(self) -> int:
        """
        Get the mode number.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The mode number.
        """
        return self.number_modes

    def set_geometric_means(self, geometric_means: np.ndarray = np.empty(0)) -> None:
        """
        Set the geometric mean.

        Parameters
        ----------
        geometric_means : np.ndarray
            The geometric mean.
            Default is an empty array.

        Returns
        -------
        None
        """
        if not isinstance(geometric_means, np.ndarray):
            geometric_means = np.array(geometric_means)
        self.parameters["geometric_means"] = geometric_means

    def get_geometric_means(self) -> np.ndarray:
        """
        Get the geometric mean.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The geometric mean.
        """
        return self.parameters["geometric_means"]

    def set_geometric_sigmas(self, geometric_sigmas: np.ndarray = np.empty(0)) -> None:
        """
        Set the geometric sigma.

        Parameters
        ----------
        geometric_sigmas : np.ndarray
            The geometric sigma.
            Default is an empty array.

        Returns
        -------
        None
        """
        if not isinstance(geometric_sigmas, np.ndarray):
            geometric_sigmas = np.array(geometric_sigmas)

        self.parameters["geometric_sigmas"] = geometric_sigmas

    def get_geometric_sigmas(self) -> np.ndarray:
        """
        Get the geometric sigma.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The geometric sigma.
        """
        return self.parameters["geometric_sigmas"]

    def set_scale_factors(self, scale_factor: np.ndarray = np.empty(0)) -> None:
        """
        Set the scale factor.

        Parameters
        ----------
        scale_factor : np.ndarray
            The scale factor.
            Default is an empty array.

        Returns
        -------
        None
        """
        if not isinstance(scale_factor, np.ndarray):
            scale_factor = np.array(scale_factor)

        self.parameters["scale_factors"] = scale_factor

    def get_scale_factors(self) -> np.ndarray:
        """
        Get the scale factor.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The scale factors.
        """
        return self.parameters["scale_factors"]

    def set_number_concentration(self) -> None:
        """
        Set the number concentration.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.parameters["number_concentration"] = np.sum(self.get_scale_factors())

    def get_number_concentration(self) -> float:
        """
        Get the number concentration.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The number concentration.
        """
        return self.parameters["number_concentration"]

    def update_model_parameters(self, params: Union[lmfit.Parameters, None] = None) -> None:
        """
        Update the model parameters of the particle size distribution. The
        default method updates the following.

        params["geometrical_means"].set(value=3.77e-06, min=0)
        params["geometrical_sigmas"].set(value=1.38e+00, min=0)
        params["scale_factors"].set(value=2.73e+08, min=0)

        Parameters
        ----------
        params : lmfit.Parameters
            The model parameters of the particle size distribution.
            Default is None.
            If None, the above explained default parameters are used.

        Returns
        -------
        None
        """

        if params == None:
            # Set the default parameters
            geomeans = lmfit.Parameter(name="geometric_means", value=3.77e-06, min=0)
            geosigs = lmfit.Parameter(name="geometric_sigmas", value=1.38e00, min=0)
            scalefacs = lmfit.Parameter(name="scale_factors", value=2.73e08, min=0)

            for param in [geomeans, geosigs, scalefacs]:
                self.update_individual_model_parameters(parameter=param)
        elif isinstance(lmfit.Parameters, params):
            self.set_model_parameters(params)
        else:
            raise TypeError("The type of the parameters must be lmfit.Parameters or None.")

    def __autoupdate_parameters__(self) -> None:
        """
        Autoupdate all dependent parameters. The default method changes
        nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            self.set_mode_number()
            self.set_number_concentration()
            # print("Autoupdate of all parameters performed.")
        except Exception as e:
            pass
            # warnings.warn(f"Autoupdate of all parameters not performed, due to error: {e}")

    def __add__(self, other: PSD_LnNormal) -> PSD_LnNormal:
        geometric_means = np.concatenate((self.get_geometric_means(), other.get_geometric_means()))
        geometric_sigmas = np.concatenate((self.get_geometric_sigmas(), other.get_geometric_sigmas()))
        scale_factors = np.concatenate((self.get_scale_factors(), other.get_scale_factors()))
        return PSD_LnNormal(
            geometric_means=geometric_means,
            geometric_sigmas=geometric_sigmas,
            scale_factors=scale_factors,
        )

    def __str__(self):
        nmodes = f"nmodes = {self.get_mode_number():.2e}"

        geomeans = ""
        geosigs = ""
        scalefacs = ""
        if self.get_mode_number() <= 0:
            return "No modes found"

        else:
            for i in range(self.get_mode_number()):
                geomeans += f"{self.get_geometric_means()[i]:.2e}, "
                geosigs += f"{self.get_geometric_sigmas()[i]:.2e}, "
                scalefacs += f"{self.get_scale_factors()[i]:.2e}, "

        geomeans = f"geomeans = [{geomeans}]"
        geosigs = f"geosigs = [{geosigs}]"
        scalefacs = f"scalefacs = [{scalefacs}]"
        numconc = f"numconc = {np.sum(self.get_number_concentration()):.2e}"
        return "\n".join([nmodes, geomeans, geosigs, scalefacs, numconc])

    def eval_func(self, radii: np.ndarray) -> np.ndarray:
        """
        Evaluate the model of the particle size distribution.

        Parameters
        ----------
        **kwargs : dict
            The keyword arguments for the model function.
            See also the lmfit.model.Model.eval.
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.eval

        Returns
        -------
        np.ndarray
            The evaluated model of the particle size distribution.
        """
        params = self.get_parameters()

        result = np.zeros_like(radii)
        for idx in range(self.get_mode_number()):
            result += self.func(
                radii=radii,
                scale_factors=params["scale_factors"][idx],
                geometric_means=params["geometric_means"][idx],
                geometric_sigmas=params["geometric_sigmas"][idx],
            )
        # For each mode, evaluate the model

        return result


def fit_lnnormal_for_psd(da: xr.DataArray, dim: str = "radius") -> PSD_LnNormal:
    """
    Fit a lognormal distribution to the PSD.

    Parameters
    ----------
    da : xr.DataArray
        The PSD to fit.
    dim : str
        The dimension to fit the PSD.
        Default is 'radius'.

    Returns
    -------
    PSD_LnNormal
        The fitted PSD.
        If there is no finite data, the PSD is not fitted and an empty PSD_LnNormal object is returned.
    """
    psd_fit = PSD_LnNormal()

    # get the dimension
    da_dim = da[dim]
    # expand the dimension to the same shape as the data
    da_dim_expanded = shape_dim_as_dataarray(da=da, output_dim=dim)

    if (np.isfinite(da)).sum() > 0:
        # do not allow the mean of the lognormal to be furter than the min and max of the data
        psd_fit.update_individual_model_parameters(
            lmfit.Parameter(
                name="geometric_means",
                min=da_dim.min().data,
                max=da_dim.max().data,
            )
        )

        result = psd_fit.get_model().fit(
            data=da.data,
            radii=da_dim_expanded.data,
            params=psd_fit.get_model_parameters(),
            nan_policy="omit",
        )
        psd_fit.lmfitParameterValues_to_dict(result.params)

    return psd_fit


def fit_2lnnormal_for_psd(da_psd: xr.DataArray, dim: str = "radius", split_value=45e-6) -> PSD_LnNormal:
    """
    Fit a lognormal distribution to the PSD.

    Parameters
    ----------
    da_psd : xr.DataArray
        The PSD to fit.
    dim : str
        The dimension to fit the PSD.
        Default is 'radius'.
    split_value : float
        The value along the dimension to split the PSD.
        If PSD is given in m, then the split radius should alon be in m.
        Default is 45e-6.

    Returns
    -------
    PSD_LnNormal
        The fitted PSD and the sum of the two fitted lognormal PSDs for the lower and upper split value.
        If there is no finite data, the PSD is not fitted and an empty PSD_LnNormal object is returned.
        If only one side of the split value has finite data, only one lognormal PSD is fitted.
    """
    # initialize the PSD
    psd_fit = PSD_LnNormal()

    da_lower = da_psd.sel(radius=slice(None, split_value))
    da_upper = da_psd.sel(radius=slice(split_value, None))

    psd_lower = fit_lnnormal_for_psd(da=da_lower, dim=dim)
    psd_upper = fit_lnnormal_for_psd(da=da_upper, dim=dim)

    psd_fit = psd_fit + psd_lower + psd_upper

    return psd_fit


def fit_linear_thermodynamics(
    da_thermo: xr.DataArray,
    thermo_fit: Union[ThermodynamicLinear, None] = None,
    dim: str = "radius",
    f0_boundaries: bool = True,
) -> ThermodynamicLinear:
    """
    Fit a linear or split linear function to the thermodynamic data.

    Parameters
    ----------
    da_thermo : xr.DataArray
        The thermodynamic data to fit.
    thermo_fit : Union[ThermodynamicLinear, None], optional
        The thermodynamic data to fit.
        If None, a new ThermodynamicLinear object is created.
        Default is None.
    dim : str, optional
        The dimension to fit the thermodynamic data.
        Default is 'radius'.
    f0_boundaries : bool, optional
        If True, the y-intercept is bounded by the min and max of the data.
        Default is True.

    Returns
    -------
    Union[ThermodynamicLinear, ThermodynamicSplitLinear]
        The fitted thermodynamic data.
        If there is no finite data, the thermodynamic data is not fitted and an empty ThermodynamicLinear or ThermodynamicSplitLinear object is returned.
    """
    # initialize the PSD
    if thermo_fit is None:
        thermo_fit = ThermodynamicLinear()

    # get the dimension
    da_dim = da_thermo[dim]
    # expand the dimension to the same shape as the data
    da_dim_expanded = shape_dim_as_dataarray(da=da_thermo, output_dim=dim)

    if (np.isfinite(da_thermo)).sum() > 0:
        if f0_boundaries == True:
            # make sure the
            thermo_fit.update_individual_model_parameters(
                lmfit.Parameter(
                    name="f_0",
                    min=da_thermo.min().data,
                    max=da_thermo.max().data,
                )
            )

        # fit the data to the model
        result = thermo_fit.get_model().fit(
            data=da_thermo.data,
            x=da_dim_expanded.data,
            params=thermo_fit.get_model_parameters(),
            nan_policy="omit",
        )
        # update the model parameters
        thermo_fit.lmfitParameterValues_to_dict(result.params)

    return thermo_fit


def fit_splitlinear_thermodynamics(
    da_thermo: xr.DataArray,
    thermo_fit: Union[ThermodynamicSplitLinear, None] = None,
    dim: str = "radius",
    f0_boundaries: bool = True,
    x_split: Union[float, None] = None,
    x_split_boundaries: bool = True,
) -> ThermodynamicSplitLinear:
    """
    Fit a linear or split linear function to the thermodynamic data.

    Parameters
    ----------
    da_thermo : xr.DataArray
        The thermodynamic data to fit.
    thermo_fit : Union[ThermodynamicLinear, None], optional
        The thermodynamic data to fit.
        If None, a new ThermodynamicLinear object is created.
        Default is None.
    dim : str, optional
        The dimension to fit the thermodynamic data.
        Default is 'radius'.
    f0_boundaries : bool, optional
        If True, the y-intercept is bounded by the min and max of the data.
        Default is True.
    x_split : float or None
        The split level of the split linear function.
        If None, the split level is not prescribed and ``x_split_boundaries`` will be evaluated.
    x_split_boundaries : bool, optional
        If True, the split level is bounded by the 150 and the max of the data.
        Default is True.

    Returns
    -------
    Union[ThermodynamicLinear, ThermodynamicSplitLinear]
        The thermodynamic fit.
        If there is no finite data, the thermodynamic data is not fitted and an empty ThermodynamicLinear or ThermodynamicSplitLinear object is returned.
    """
    # initialize the PSD
    if thermo_fit is None:
        thermo_fit = ThermodynamicSplitLinear()

    # get the dimension
    da_dim = da_thermo[dim]
    # expand the dimension to the same shape as the data
    da_dim_expanded = shape_dim_as_dataarray(da=da_thermo, output_dim=dim)

    if (np.isfinite(da_thermo)).sum() > 0:
        if f0_boundaries == True:
            # make sure the
            thermo_fit.update_individual_model_parameters(
                lmfit.Parameter(
                    name="f_0",
                    min=da_thermo.min().data,
                    max=da_thermo.max().data,
                )
            )
        if x_split is not None:
            # make sure the split level is the prescribed value
            thermo_fit.update_individual_model_parameters(
                lmfit.Parameter(
                    name="x_split",
                    value=x_split,
                    vary=False,
                )
            )
        elif x_split_boundaries == True:
            # only allow the split level to be between 150m and max of the data
            thermo_fit.update_individual_model_parameters(
                lmfit.Parameter(
                    name="x_split",
                    min=150,
                    max=float(da_dim.max()),
                )
            )

        # fit the data to the model
        result = thermo_fit.get_model().fit(
            data=da_thermo.data,
            x=da_dim_expanded.data,
            params=thermo_fit.get_model_parameters(),
            nan_policy="omit",
        )
        # update the model parameters
        thermo_fit.lmfitParameterValues_to_dict(result.params)

    return thermo_fit


def fit_thermodynamics(
    da_thermo: xr.DataArray,
    thermo_fit: Union[ThermodynamicSplitLinear, ThermodynamicLinear],
    dim: str = "radius",
    f0_boundaries: bool = True,
    x_split: Union[float, None] = None,
    x_split_boundaries: bool = True,
) -> Union[ThermodynamicSplitLinear, ThermodynamicLinear]:
    """
    Fit a linear or split linear function to the thermodynamic data.

    Parameters
    ----------
    da_thermo : xr.DataArray
        The thermodynamic data to fit.
    thermo_fit : Union[ThermodynamicLinear, ThermodynamicSplitLinear]
        The thermodynamic fit type.
        This can be either ThermodynamicLinear or ThermodynamicSplitLinear.
    dim : str, optional
        The dimension to fit the thermodynamic data.
        Default is 'radius'.
    f0_boundaries : bool, optional
        If True, the y-intercept is bounded by the min and max of the data.
        Default is True.
    x_split_boundaries : bool, optional
        If True, the split level is bounded by 150 min and max of the dimension.
        Default is True.

    Returns
    -------
    Union[ThermodynamicSplitLinear, ThermodynamicLinear]
        The thermodynamic fit .
    """

    if isinstance(thermo_fit, ThermodynamicSplitLinear):
        fit_func = fit_splitlinear_thermodynamics(
            da_thermo=da_thermo,
            thermo_fit=thermo_fit,
            dim=dim,
            f0_boundaries=f0_boundaries,
            x_split=x_split,
            x_split_boundaries=x_split_boundaries,
        )
    elif isinstance(thermo_fit, ThermodynamicLinear):
        fit_func = fit_linear_thermodynamics(
            da_thermo=da_thermo,
            thermo_fit=thermo_fit,
            dim=dim,
            f0_boundaries=f0_boundaries,
        )
    else:
        raise TypeError(
            "The type of the input must be either ThermodynamicLinear or ThermodynamicSplitLinear."
        )

    return fit_func
