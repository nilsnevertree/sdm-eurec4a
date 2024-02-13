"""Module to transfer the fitted observations to CLEO."""
from __future__ import annotations

import warnings

from typing import Dict, Tuple, Union

import lmfit
import numpy as np
import yaml

from scipy.optimize import curve_fit
from sdm_eurec4a.input_processing.models import lnnormaldist, split_linear_func


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
