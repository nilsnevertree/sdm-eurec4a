import warnings

import numpy as np
import xarray as xr


def vsd_from_psd(
    ds: xr.Dataset,
    psd_name: str = "particle_size_distribution",
    psd_factor: float = 1,
    scale_name: str = "radius",
    scale_factor: float = 1,
    radius_given: bool = True,
) -> xr.DataArray:
    """
    Calculate the volume size distribution from the particle size distribution.

    VSD = PSD * 4/3 * pi * r^3

    Note
    ----
    - If the diameter is given, set radius to False.
    - Make sure, the units are correct.
    - If PSD is normalized by bin width (given in .../µm), the VSD is normalized by bin width (given in .../µm).
    - For instance given are:
        - PSD in #/l (#/(dm^3))
        - Diameter in µm (1e-6m)
    - Then set:
        - psd_factor to 1e-6
        - scale_factor to 1e-6
        - radius_given = False

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the particle size distribution (``psd_name``) and the diameter or radius (``scale_name``).
    psd_name : str, optional
        The name of the particle size distribution. Default is "particle_size_distribution".
        Assumed to have units of 1/m^3.
    psd_factor : float, optional
        The factor to convert the PSD to the correct units. Default is 1.
    scale_name : str, optional
        The name of the scale. Default is "diameter".
    scale_factor : float, optional
        The scale factor. Default is 1.
    radius_given : bool, optional
        If set to True, it is assumed, that under ``scale_name`` the radius is stored.
        If set to False, it is assumed, that under ``scale_name`` the diameter is stored.
        Default is True.

    Returns
    -------
    xr.DataArray
        The volume size distribution.
        Make sure to check the units and otherwise set the value!
    """
    psd = ds[psd_name] * psd_factor

    if radius_given == True:
        radius = ds[scale_name] * scale_factor
    else:
        radius = 0.5 * ds[scale_name] * scale_factor

    vsd = psd * 4 / 3 * np.pi * radius**3
    vsd.attrs = dict(
        units="m^3/m^3",
        long_name="Volume size distribution",
        description="Volume size distribution calculated from the particle size distribution. VSD = PSD * 4/3 * pi * radius^3",
    )
    vsd.name = "volume_size_distribution"
    # warnings.warn("units is set to m^3/m^3. Make sure to check the units and otherwise set the value!")
    return vsd


def msd_from_psd(
    ds: xr.Dataset,
    psd_name: str = "particle_size_distribution",
    psd_factor: float = 1,
    scale_name: str = "radius",
    scale_factor: float = 1,
    rho_water: float = 1000,
    radius_given: bool = True,
) -> xr.DataArray:
    """
    Calculate the mass size distribution from the particle size distribution.
    A constant density of water droplets is assumed to be rho_water = 1000 kg/m^3.
    Spherical droplets are assumed.

    MSD = rho_water * PSD * 4/3 * pi * r^3
        = rho_water * VSD

    It uses the vsd_from_psd function.

    Note
    ----
    - If the diameter is given, set radius to False. (default)
    - Make sure, the units are correct!!!
        If PSD is normalized by bin width (given in .../µm),
        the MSD is normalized by bin width (given in .../µm).
    - For instance given are:
        - PSD in #/l (#/(dm^3))
        - Diameter in µm (1e-6m)
    - Then set:
        - psd_factor to 1e-6
        - scale_factor to 1e-6
        - radius_given = False

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the particle size distribution (``psd_name``) and the diameter or radius (``scale_name``).
    psd_name : str, optional
        The name of the particle size distribution. Default is "particle_size_distribution".
        Assumed to have units of 1/m^3.
    psd_factor : float, optional
        The factor to convert the PSD to the correct units. Default is 1.
    scale_name : str, optional
        The name of the scale. Default is "radius".
    scale_factor : float, optional
        The scale factor. Default is 1.
    rho_water : float, optional
        density of water, by default 1000 kg/m^3
    radius_given : bool, optional
        If set to True, it is assumed, that under ``scale_name`` the radius is stored.
        If set to False, it is assumed, that under ``scale_name`` the diameter is stored.
        Default is True.

    Returns
    -------
    xr.DataArray
        The mass size distribution in kg/m^3.
        Make sure to check the units and otherwise set the value!
    """

    vsd = vsd_from_psd(
        ds=ds,
        psd_name=psd_name,
        psd_factor=psd_factor,
        scale_name=scale_name,
        scale_factor=scale_factor,
        radius_given=radius_given,
    )
    msd = rho_water * vsd
    msd.attrs = dict(
        units="kg/m^3",
        long_name="Mass size distribution",
        description="Mass size distribution calculated from the particle size distribution. MSD = rho_water * PSD * 4/3 * pi * radius^3",
    )
    msd.name = "mass_size_distribution"
    # warnings.warn("units is set to kg/m^3. Make sure to check the units and otherwise set the value!")
    return msd


def lwc_from_psd(
    ds: xr.Dataset,
    sum_dim: str = "radius",
    psd_name: str = "particle_size_distribution",
    psd_factor: float = 1,
    scale_name: str = "radius",
    scale_factor: float = 1,
    rho_water: float = 1000,
    radius_given: bool = True,
) -> xr.DataArray:
    """
    Calculate the liquid water content from the particle size distribution.
    A constant density of water droplets is assumed to be rho_water = 1000 kg/m^3.
    Spherical droplets are assumed.

    LWC = sum over all diameters of (rho_water * PSD * 4/3 * pi * r^3)
        = sum over all diameters of * MSD

    It uses the msd_from_psd function.

    Note
    ----
    - If the diameter is given, set radius to False. (default)
    - Make sure, the units are correct!!!
        If PSD is normalized by bin width (given in .../µm),
        the LWC is not correct!!!
    - For instance given are:
        - PSD in #/l (#/(dm^3))
        - Diameter in µm (1e-6m)
    - Then set:
        - psd_factor to 1e-6
        - scale_factor to 1e-6
        - radius_given = False

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the particle size distribution (``psd_name``) and the diameter or radius (``scale_name``).
    sum_dim : str, optional
        The dimension to sum over. Default is "radius".
        This allows to sum over all diameters to get the liquid water content for each value of the other dimension.
    psd_name : str, optional
        The name of the particle size distribution. Default is "particle_size_distribution".
        Assumed to have units of 1/m^3.
    psd_factor : float, optional
        The factor to convert the PSD to the correct units. Default is 1.
    scale_name : str, optional
        The name of the scale. Default is "radius".
    scale_factor : float, optional
        The scale factor. Default is 1.
    rho_water : float, optional
        density of water, by default 1000 kg/m^3
    radius_given : bool, optional
        If set to True, it is assumed, that under ``scale_name`` the radius is stored.
        If set to False, it is assumed, that under ``scale_name`` the diameter is stored.
        Default is True.

    Returns
    -------
    xr.DataArray
        The total liquid water content in kg/m^3.
        Make sure to check the units and otherwise set the value!
    """

    msd = msd_from_psd(
        ds=ds,
        psd_name=psd_name,
        psd_factor=psd_factor,
        scale_name=scale_name,
        scale_factor=scale_factor,
        radius_given=radius_given,
        rho_water=rho_water,
    )
    lwc = msd.sum(dim=sum_dim)
    lwc.attrs = dict(
        units="kg/m^3",
        long_name="Liquid water content",
        description="Liquid water content calculated from the particle size distribution. LWC is the sum over specified dimension of (rho_water * PSD * 4/3 * pi * radius^3)",
    )
    lwc.name = "liquid_water_content"
    # warnings.warn("units is set to kg/m^3. Make sure to check the units and otherwise set the value!")
    return lwc


def saturation_vapour_pressure(temperature: np.ndarray) -> np.ndarray:
    """
    Calculate the saturation vapour pressure over water for a given
    temperature.

    Parameters
    ----------
    temperature : np.ndarray
        The temperature in Kelvin.

    Returns
    -------
    np.ndarray
        The saturation vapour pressure in Pa.
    """
    T = temperature
    A_w = 2.543e11  # Pa
    B_w = 5420  # K
    es = A_w * np.exp(-B_w / T)
    es = __rename_if_dataarray__(es, "saturation_vapour_pressure")
    return es


def water_vapour_pressure(
    specific_humidity: np.ndarray, pressure: np.ndarray, simplified: bool = False
) -> np.ndarray:
    """
    Calculate the water vapour pressure from the specific humidity and the
    pressure. This follows (2.80) from Introduction to Clouds: From the
    Microscale to Climate.

    Simplified version uses:
    q_v = (epsilon * e) /  p
    e = q_v * p / epsilon

    Non simplified version uses:
    q_v = (epsilon * e) / (p - e + epsilon * e)
    e = q_v * p / (epsilon + q_v - epsilon * q_v)


    Citation
    --------
    (2.80) from Introduction to Clouds: From the Microscale to Climate
    Ulrike Lohmann, Felix Lüönd, Fabian Mahrt, and Gregor Feingold
    ISBN: 978-1-107-01822-8 978-1-139-08751-3


    Parameters
    ----------
    specific_humidity : np.ndarray
        The specific humidity in kg/kg.
    pressure : np.ndarray
        The pressure in Pa.

    Returns
    -------
    np.ndarray
        The vapour pressure in Pa.
    """
    q_v = specific_humidity
    p = pressure
    epsilon = 0.622
    if simplified:
        e = q_v * p / epsilon
    else:
        e = q_v * p / (epsilon + q_v - epsilon * q_v)
    return e


def relative_humidity(saturation_vapour_pressure: np.ndarray, vapour_pressure: np.ndarray) -> np.ndarray:
    """
    Calculate the relative humidity from the saturation vapour pressure and the
    vapour pressure.

    Parameters
    ----------
    saturation_vapour_pressure : np.ndarray
        The saturation vapour pressure in Pa.
    vapour_pressure : np.ndarray
        The vapour pressure in Pa.

    Returns
    -------
    np.ndarray
        The relative humidity in %.
    """
    rh = 100 * vapour_pressure / saturation_vapour_pressure
    rh = __rename_if_dataarray__(rh, "relative_humidity")
    return rh


def relative_humidity_from_tps(
    temperature: np.ndarray,
    pressure: np.ndarray,
    specific_humidity: np.ndarray,
    simplified: bool = False,
):
    """
    Calculate the relative humidity from the temperature, pressure and specific
    humidity.

    Parameters
    ----------
    temperature : np.ndarray
        The temperature in Kelvin.
    pressure : np.ndarray
        The pressure in Pa.
    specific_humidity : np.ndarray
        The specific humidity in kg/kg.
    simplified : bool, optional
        If set to True, the simplified version is used.
        Default is False.

    Returns
    -------
    np.ndarray
        The relative humidity in %.
    """
    es = saturation_vapour_pressure(temperature)
    e = water_vapour_pressure(specific_humidity, pressure, simplified)
    rh = relative_humidity(es, e)
    rh = __rename_if_dataarray__(rh, "relative_humidity")
    return rh


def __rename_if_dataarray__(da, name):
    if isinstance(da, xr.DataArray):
        print("rename dataarray")
        da.name = name
    return da
