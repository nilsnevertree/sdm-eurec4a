"""Module to transfer the fitted observations to CLEO."""
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
from typing import Union
import lmfit
from sdm_eurec4a.input_processing.models import lnnormaldist

class Transfer:
    """Class to transfer the fitted observations to CLEO."""
    def __init__(self) -> None:
        """Initialize the Transfer class."""
        self.thermodynamic = None
        self.particle_size_distribution = None

    def transfer_thermodynamic(self, thermodynamic: dict) -> None:
        """Transfer the fitted thermodynamic observations to CLEO."""
        self.thermodynamic = thermodynamic

    def transfer_particle_size_distribution(self, particle_size_distribution: dict) -> None:
        """Transfer the fitted particle size distribution observations to CLEO."""
        self.particle_size_distribution = particle_size_distribution

    def transfer(self) -> None:
        """Transfer the fitted observations to CLEO."""
        pass


class TransferError(Exception):
    """Exception raised for errors in the Transfer class."""
    def __init__(self, message: str) -> None:
        """Initialize the TransferError class."""
        self.message = message
        super().__init__(self.message)

class ThermodynamicInput():
    """Class to handle the thermodynamic input."""
    def __init__(self) -> None:
        """Initialize the ThermodynamicInput class."""
        self.thermodynamic = None

    def set_thermodynamic(self, thermodynamic: dict) -> None:
        """Set the thermodynamic input."""
        self.thermodynamic = thermodynamic

    def get_thermodynamic(self) -> dict:
        """Get the thermodynamic input."""
        return self.thermodynamic
    


class ParticleSizeDistributionInput():
    """Class to handle the particle size distribution input."""
    def __init__(
            self, 
            type : str = "None", 
            func : function = lambda x : None, 
            independent_vars : list = [],
            parameters :dict = {}, 
            model_arguments : dict = {},
            ) -> None:
        """Initialize the ParticleSizeDistributionInput class."""
        self.particle_size_distribution = None
        self.independent_vars = independent_vars
        self.set_parameters(parameters=parameters)
        self.set_type(type=type)
        self.set_func(func=func)
        self.set_model(**model_arguments)

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
        self.type : str = type

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

    def set_func(self, func: function = lambda x : None) -> None:
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
        self.func : function = func

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
    
    def set_model(self, **kwargs : dict()) -> None:
        """
        Set the model of the particle size distribution.
        It uses the lmfit.Model class.
        The function to be used for this is the function set in the set_func method.
        The function cannot be provided as a parameter to this method!
        It is set in the set_func method.

        The Model can be used to fit the particle size distribution to the data.

        
        Parameters
        ----------
        **kwargs : dict
            The keyword arguments for the lmfit.Model class.
            See also the lmfit.Model class.
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model

        Returns
        -------
        None
        """
        self.model : lmfit.Model = lmfit.Model(func = self.get_func(), independent_vars= self.independent_vars, **kwargs)
        # make sure that radii is the independent variable
        self.__set_model_independ_vars__()
        self.__update_model__()
        self.set_model_parameters(self.model.make_params())
        self.__update_model_parameters__()

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
        Update the model of the particle size distribution.
        The default method changes nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def get_model(self) -> lmfit.Model:
        """
        Get the model of the particle size distribution.
        
        Parameters
        ----------
        None

        Returns
        -------
        lmfit.Model
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

    def __update_model_parameters__(self) -> None:
        """
        Update the model parameters of the particle size distribution.
        The default method changes nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def fit_model(
        self, 
        data, 
        params=None,
        weights=None, 
        method='leastsq',
        iter_cb=None, 
        scale_covar=True, 
        verbose=False, 
        fit_kws=None,
        nan_policy=None, 
        calc_covar=True, 
        max_nfev=None,
        coerce_farray=True, 
        **kwargs) -> lmfit.ModelResult:
        """
        This method fits the model to the data using the supplied Parameters.
        The docstring was copied from  https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.fit

        This function will return a ModelResult object which is the same as under ``self.model_result``.
        It also overwrites the self.parameters with the fitted parameters!

        Notes
        -----
        1. If params is None, the values under ``self.model_parameters`` are used. This is the prefered way to handle this!
        2. All non-parameter arguments for the model function, **including all the independent variables** will need to be passed in using keyword arguments.
        3. This function will return a ModelResult object which is the same as under ``self.model_result``.

        **Description as in the original docstring:**

        Fit the model to the data using the supplied Parameters.

        Parameters
        ----------
        data : array_like
            Array of data to be fit.
        params : Parameters, optional
            Parameters to use in fit (default is None).
        weights : array_like, optional
            Weights to use for the calculation of the fit residual [i.e.,
            `weights*(data-fit)`]. Default is None; must have the same size as
            `data`.
        method : str, optional
            Name of fitting method to use (default is `'leastsq'`).
        iter_cb : callable, optional
            Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix when
            calculating uncertainties (default is True).
        verbose : bool, optional
            Whether to print a message when a new parameter is added
            because of a hint (default is True).
        fit_kws : dict, optional
            Options to pass to the minimizer being used.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than `'leastsq'` and `'least_squares'`.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        coerce_farray : bool, optional
            Whether to coerce data and independent data to be ndarrays
            with dtype of float64 (or complex128).  If set to False, data
            and independent data are not coerced at all, but the output of
            the model function will be. (default is True)
        **kwargs : optional
            Arguments to pass to the model function, possibly overriding
            parameters.

        Returns
        -------
        ModelResult

        Notes
        -----
        1. if `params` is None, the values for all parameters are expected
        to be provided as keyword arguments. Mixing `params` and
        keyword arguments is deprecated (see `Model.eval`).

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        3. Parameters are copied on input, so that the original Parameter objects
        are unchanged, and the updated values are in the returned `ModelResult`.

        Examples
        --------
        Take ``t`` to be the independent variable and data to be the curve
        we will fit. Use keyword arguments to set initial guesses:

        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        Or, for more control, pass a Parameters object.

        >>> result = my_model.fit(data, params, t=t)

        """

        # Use the model parameters if no parameters are provided
        if params == None:
            params = self.model_parameters
        
        # Fit the model to the data
        self.model_result = self.model.fit(
            data = data,
            params = params,
            weights = weights,
            method = method,
            iter_cb = iter_cb,
            scale_covar = scale_covar,
            verbose = verbose,
            fit_kws = fit_kws,
            nan_policy = nan_policy,
            calc_covar = calc_covar,
            max_nfev = max_nfev,
            coerce_farray = coerce_farray,
            **kwargs
        )

        self.set_model_parameters(self.model_result.params)
        
        # Overwrite the parameters with the fitted parameters
        self.set_parameters(self.model_result.best_values)
        

        return self.model_result


class PSD_LnNormal(ParticleSizeDistributionInput):
    """
    Class to handle the particle size distribution input.
    It uses a lognormal distribution.
    """
    def __init__(
            self, 
            geometric_means: np.ndarray = np.empty(0), 
            geometric_sigmas: np.ndarray = np.empty(0), 
            scale_factors: np.ndarray = np.empty(0)
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


        Attributes
        ----------
        parameters : dict
            The parameters of the particle size distribution.
        type : str
            The type of the particle size distribution.

        """
        super().__init__(
            type = "LnNormal",
            func = lnnormaldist,
            independent_vars = ["radii"],
            parameters=dict(),
        )

        if len(geometric_means) != len(geometric_sigmas) != len(scale_factors):
            message = "The length of the geometric means, geometric sigmas, and scale factors must be the same."
            message += f"\nBut is {len(geometric_means)}, {len(geometric_sigmas)}, and {len(scale_factors)}."
            raise ValueError(message)
        else:
            self.set_geometric_means(geometric_means)
            self.set_geometric_sigmas(geometric_sigmas)
            self.set_scale_factors(scale_factors)
            self.set_mode_number()
            self.set_number_concentration()




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

    def set_geometric_sigmas(self, geometric_sigma: np.ndarray = np.empty(0)) -> None:
        """
        Set the geometric sigma.
        
        Parameters
        ----------
        geometric_sigma : np.ndarray
            The geometric sigma.
            Default is an empty array.

        Returns
        -------
        None
        """
        if not isinstance(geometric_sigma, np.ndarray):
            geometric_sigma = np.array(geometric_sigma)
        
        self.parameters["geometric_sigma"] = geometric_sigma

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
        return self.parameters["geometric_sigma"]

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
        return  self.parameters["scale_factors"]

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

    def __update_model_parameters__(self, params : Union(lmfit.Parameters, None) = None) -> None:
        """
        Update the model parameters of the particle size distribution.
        The default method updates the following

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
            geosigs = lmfit.Parameter(name="geometric_sigmas", value=1.38e+00, min=0)
            scalefacs = lmfit.Parameter(name="scale_factors", value=2.73e+08, min=0)

            for param, name in zip([geomeans, geosigs, scalefacs], ["geometrical_means", "geometrical_sigmas", "scale_factors"]):
                self.__update_individual_model_parameters__(parameter=param, name=name)
        elif isinstance(lmfit.Parameters, params):
            self.set_model_parameters(params)
        else :
            raise TypeError("The type of the parameters must be lmfit.Parameters or None.")
            
    def __update_individual_model_parameters__(self, parameter : lmfit.Parameter, name : str) -> None:
        """
        Update a specific model parameter.

        Parameters
        ----------
        parameter : lmfit.Parameters
            The model parameters of the particle size distribution.
        name : str
            The name of the parameter to be updated.
            
        Returns
        -------
        None
        """

        self.model_parameters[name] = parameter


    def __add__(self, other : PSD_LnNormal) -> PSD_LnNormal:
        
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

        else :
            for i in range(self.get_mode_number()):
                geomeans += f"{self.get_geometric_means()[i]:.2e}, "
                geosigs += f"{self.get_geometric_sigmas()[i]:.2e}, "
                scalefacs += f"{self.get_scale_factors()[i]:.2e}, "

        geomeans = f"geomeans = [{geomeans}]"
        geosigs = f"geosigs = [{geosigs}]"
        scalefacs = f"scalefacs = [{scalefacs}]"
        numconc = f"numconc = {np.sum(self.get_number_concentration()):.2e}"
        return "\n".join([nmodes, geomeans, geosigs, scalefacs, numconc])

