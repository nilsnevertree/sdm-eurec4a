"""Module to transfer the fitted observations to CLEO."""
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
from typing import Union
from lmfit import Model as lmfitModel
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
            parameters :dict = {}, 
            statistics : dict = {},
            model_arguments : dict = {},
            ) -> None:
        """Initialize the ParticleSizeDistributionInput class."""
        self.particle_size_distribution = None
        self.set_parameters(parameters=parameters)
        self.set_type(type=type)
        self.set_statistics(statistics=statistics)
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
        self.type = type

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

    def set_statistics(self, statistics: dict = dict()) -> None:
        """
        Set the statistics of the particle size distribution.
        
        Parameters
        ----------
        statistics : dict
            The statistics of the particle size distribution.
            Default is an empty dictionary.

        Returns
        -------
        None
        """
        self.statistics = statistics

    def get_statistics(self) -> dict:
        """
        Get the statistics of the particle size distribution.
        
        Parameters
        ----------
        None

        Returns
        -------
        dict
            The statistics of the particle size distribution.
        """
        return self.statistics

    def set_model(self, **kwargs) -> None:
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
        self.model = lmfitModel(func = self.get_func(), **kwargs)
        self.model_arguments = kwargs

    def get_model(self) -> function:
        """
        Get the model of the particle size distribution.
        
        Parameters
        ----------
        None

        Returns
        -------
        function
            The model of the particle size distribution.
        """
        return self.model
    
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
        self.func = func

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
            func = lnnormaldist
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

    def set_geometric_means(self, geometric_mean: np.ndarray = np.empty(0)) -> None:
        """
        Set the geometric mean.
        
        Parameters  
        ----------
        geometric_mean : np.ndarray
            The geometric mean.
            Default is an empty array.

        Returns
        -------
        None
        """
        if not isinstance(geometric_mean, np.ndarray):
            geometric_mean = np.array(geometric_mean)
        self.parameters["geometric_mean"] = geometric_mean

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
        return self.parameters["geometric_mean"]

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

    def __curve_fit__(self, xdata: np.ndarray, ydata: np.ndarray, **kwargs):
        popt, pcov = curve_fit(f=self.lnnormaldist, xdata=xdata, ydata=ydata, **kwargs)
        return popt, pcov

    def fit_parameters(self, xdata: np.ndarray, ydata: np.ndarray, **kwargs) -> Tuple:
        popt, pcov = self.__curve_fit__(xdata=xdata, ydata=ydata, **kwargs)

        self.set_geometric_means([popt[1]])
        self.set_geometric_sigmas([popt[2]])
        self.set_scale_factors([popt[0]])
        self.pcov = pcov
        return self
