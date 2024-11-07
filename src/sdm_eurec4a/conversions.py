import warnings

from typing import Union

import numpy as np
import xarray as xr


def __attrs_if_dataarray__(da: xr.DataArray, attrs: dict = {}, name: str = "") -> xr.DataArray:
    if isinstance(da, xr.DataArray):
        name = attrs.pop("name", name)
        da.attrs.update(attrs)
        da.name = name
        return da
    else:
        return da


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
        units="m^3 m^{-3}",
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
    Calculate the mass size distribution from the particle size distribution. A constant
    density of water droplets is assumed to be rho_water = 1000 kg/m^3. Spherical
    droplets are assumed.

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
        units="kg m^{-3}",
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
    Calculate the liquid water content from the particle size distribution. A constant
    density of water droplets is assumed to be rho_water = 1000 kg/m^3. Spherical
    droplets are assumed.

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
        units="kg m^{-3}",
        long_name="Liquid water content",
        description="Liquid water content calculated from the particle size distribution. LWC is the sum over specified dimension of (rho_water * PSD * 4/3 * pi * radius^3)",
    )
    lwc.name = "liquid_water_content"
    # warnings.warn("units is set to kg/m^3. Make sure to check the units and otherwise set the value!")
    return lwc


def msd_from_psd_dataarray(
    da: xr.DataArray,
    radius_name: str = "radius",
    radius_scale_factor: float = 1,
    rho_water: float = 1000,
) -> xr.DataArray:
    """
    Calculate the mass size distribution from the particle size distribution.
    A constant density of water droplets is assumed to be rho_water = 1000 kg/m^3.
    And spherical droplets are assumed.
    The input array should contain the radius or diameter information and coordinate.
    It the diameter is given, use the `radius_scale_factor` and set it to `0.5`.
    This function does not validate the units!

    MSD = rho_water * PSD * 4/3 * pi * r^3

    Note
    ----
    - If PSD is normalized by bin width (given in .../m), the MSD is normalized by bin width (given in .../m).
    - For instance given are:
        - Diameter as coordinate in µm (1e-6m)
    - Then set:
        - scale_factor = 0.5 * 1e-6

    Parameters
    ----------
    ds : xr.DataArray
        DataArray containing the particle size distribution and the radius as coordinate (``radius_name``).
    radius_name : str, optional
        The name of the radius coordinate.
        Default is "radius".
        If the radius is not given in SI units, make sure to adapt the `radius_scale_factor` appropriately.
        If for instance the diameter is given, set it to "diameter" and set `radius_scale_factor` to 0.5.
    radius_scale_factor : float, optional
        Factor by which the radius is scaled. Default is 1.
        For instance if the radius is given in µm, set it to 1e6.
    rho_water : float, optional
        density of water, by default 1000 kg/m^3

    Returns
    -------
    xr.DataArray
        The mass size distribution in kg/m^3.
        Make sure to check the units and otherwise set the value!
    """

    # make sure the radius is given in the correct units
    radius = da[radius_name] * radius_scale_factor

    msd = rho_water * da * 4 / 3 * np.pi * radius**3
    msd.attrs = dict(
        units="kg m^{-3}",
        long_name="Mass size distribution",
        description="Mass size distribution calculated from the particle size distribution. MSD = rho_water * PSD * 4/3 * pi * radius^3",
        comment="Make sure to check the units and otherwise set the value!",
    )
    return msd


def psd_from_msd_dataarray(
    da: xr.DataArray,
    radius_name: str = "radius",
    radius_scale_factor: float = 1,
    rho_water: float = 1000,
) -> xr.DataArray:
    """
    Inverse of msd_from_psd_dataarray.

    Parameters
    ----------
    ds : xr.DataArray
        DataArray containing the mass size distribution and the radius as coordinate (``radius_name``).
    radius_name : str, optional
        The name of the radius coordinate.
        Default is "radius".
        If the radius is not given in SI units, make sure to adapt the `radius_scale_factor` appropriately.
        If for instance the diameter is given, set it to "diameter" and set `radius_scale_factor` to 0.5.
    radius_scale_factor : float, optional
        Factor by which the radius is scaled. Default is 1.
        For instance if the radius is given in µm, set it to 1e6.
    rho_water : float, optional
        density of water, by default 1000 kg/m^3

    Returns
    -------
    xr.DataArray
        The particle size distribution in 1/m^3.
        Make sure to check the units and otherwise set the value!
    """

    # make sure the radius is given in the correct units
    radius = da[radius_name] * radius_scale_factor

    psd = da * radius ** (-3) / (rho_water * 4 / 3 * np.pi)

    psd.attrs = dict(
        units="m^{-3}",
        long_name="Particle size distribution",
        description="Particle size distribution calculated from the mass size distribution. PSD = MSD / (rho_water * 4/3 * pi * radius^3)",
        comment="Make sure to check the units and otherwise set the value!",
    )
    return psd


def saturation_vapour_pressure(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate the saturation vapour pressure over water for a given temperature.

    Parameters
    ----------
    temperature : xr.DataArray
        The temperature in Kelvin.

    Returns
    -------
    np.ndarray
        The saturation vapour pressure in Pa.
    """
    T = temperature
    A_w = 2.543e11  # Pa
    B_w = 5420  # K
    es: xr.DataArray = A_w * np.exp(-B_w / T)  # Pa (K/K) = Pa #type: ignore
    attrs = dict(
        name="saturation_vapour_pressure",
        units="Pa",
        long_name="Saturation vapour pressure",
        description="Saturation vapour pressure over water calculated from temperature.",
    )
    es = __attrs_if_dataarray__(da=es, attrs=attrs)
    return es


def water_vapour_pressure(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
    simplified: bool = False,
) -> xr.DataArray:
    """
    Calculate the water vapour pressure from the specific humidity and the pressure.
    This follows (2.80) from Introduction to Clouds: From the Microscale to Climate.

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
    specific_humidity : xr.DataArray
        The specific humidity in kg/kg.
    pressure : xr.DataArray
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


def relative_humidity(
    saturation_vapour_pressure: xr.DataArray,
    vapour_pressure: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the relative humidity from the saturation vapour pressure and the vapour
    pressure.

    Parameters
    ----------
    saturation_vapour_pressure : xr.DataArray
        The saturation vapour pressure in Pa.
    vapour_pressure : xr.DataArray
        The vapour pressure in Pa.

    Returns
    -------
    np.ndarray
        The relative humidity in %.
    """
    rh: xr.DataArray = 100 * vapour_pressure / saturation_vapour_pressure  # type: ignore
    attrs = dict(
        name="relative_humidity",
        units=r"\%",
        long_name="Relative humidity",
        description="Relative humidity calculated from saturation vapour pressure and vapour pressure.",
    )
    rh = __attrs_if_dataarray__(da=rh, attrs=attrs)
    return rh


def saturation_vapour_pressure_murphy_koop(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate the saturation vapour pressure according to Murphy and Koop (2005)

    Parameters
    ----------
    temperature : xr.DataArray
        Temperature in Kelvin

    Returns
    -------
    xr.DataArray
        Saturation vapour pressure in Pa

    """

    innest = 53.878 - 1331.22 / temperature - 9.44523 * np.log(temperature) + 0.014025 * temperature
    inner = 54.842763 - 6763.22 / temperature - 4.210 * np.log(temperature) + 0.000367 * temperature

    es: xr.DataArray = np.exp(inner + np.tanh(0.0415 * (temperature - 218.8)) * innest)  # type: ignore
    es = __attrs_if_dataarray__(
        es,
        attrs=dict(
            name="saturation_vapour_pressure",
            unit="Pa",
            long_name="Saturation vapour pressure",
            comment="Follows the equation of Murphy and Koop (2005) given in Lohmann et al. (2016)",
        ),
    )
    return es


def saturation_vapour_pressure_Bolton(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate the saturation vapour pressure according to Bolton (1980)
    """
    T = temperature - 273.15
    es: xr.DataArray = 611.2 * np.exp(17.67 * T / (T + 243.5))  # type: ignore
    es = __attrs_if_dataarray__(
        es,
        attrs=dict(
            name="saturation_vapour_pressure",
            unit="Pa",
            long_name="Saturation vapour pressure",
        ),
    )

    return es


def saturation_vapour_pressure_Roger_Yau(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate the saturation vapour pressure according to Rogers and Yau (1989)
    """

    A_w = 2.543e11  # Pa
    B_w = 5420  # K
    es: xr.DataArray = A_w * np.exp(-B_w / temperature)  # Pa (K/K) = Pa # type: ignore
    es = __attrs_if_dataarray__(
        es,
        attrs=dict(
            name="saturation_vapour_pressure",
            unit="Pa",
            long_name="Saturation vapour pressure",
        ),
    )
    return es


def partial_pressure(
    temperature: xr.DataArray, partial_density: xr.DataArray, specific_gas_constant: float
) -> xr.DataArray:
    """
    Calculate the partial pressure of an ideal gas.
    For instance the partial pressure of water vapour in the atmosphere.

    Parameters
    ----------
    temperature : xr.DataArray
        Temperature in Kelvin
    partial_density : xr.DataArray
        Partial density of the gas in kg/m^3
    specific_gas_constant : float
        Specific gas constant of the gas in J/(kg K)
        For water vapour this is 461.5 J/(kg K)
        For dry air this is 287.05 J/(kg K)

    Returns
    -------
    xr.DataArray
        Partial pressure in Pa

    Examples
    --------
    >>> # Calculate the partial pressure of water vapour in the atmosphere.
    >>> # The partial density of water vapour is 0.01 kg/m^3
    >>> # The temperature is 300 K
    >>> # The specific gas constant of water vapour is 461.5 J/(kg K)
    >>> partial_pressure(temperature = 300, partial_density = 0.01, specific_gas_constant = 461.5)
    ... 1384.5
    """
    return partial_density * specific_gas_constant * temperature


def relative_humidity_partial_density(
    temperature: xr.DataArray, partial_density: xr.DataArray, specific_gas_constant: float = 461.5
) -> xr.DataArray:
    """
    Calculate the relative humidity of the atmosphere from:
    - the temperature in Kelvin
    - the partial density of the water vapour in kg/m^3
    - the specific gas constant of the water vapour in J/(kg K)

    The caculation is based on the ideal gas law and the saturation vapour pressure.
    First the partial pressure of the water vapour is calculated.
    Then the saturation vapour pressure is calculated based on the definition by Murphy and Koop (2005).

    e = \\rho_w R_w T
    e_{sat} = 54.842763 - 6763.22 / T - 4.210 * \\log(T) + 0.000367 * T
    RH = \\frac{e}{e_{sat}} * 100

    Parameters
    ----------
    temperature : xr.DataArray
        Temperature in Kelvin
    partial_density : xr.DataArray
        Partial density of the water vapour in kg/m^3
    specific_gas_constant : float
        Specific gas constant of the water vapour in J/(kg K)
        The default value is 461.5 J/(kg K) as given in Lohmann et al. (2016)

    Returns
    -------
    xr.DataArray
        Relative humidity in %

    Examples
    --------
    >>> # Calculate the relative humidity of the atmosphere.
    >>> # The partial density of water vapour is 0.01 kg/m^3
    >>> # The temperature is 300 K
    >>> # The specific gas constant of water vapour is 461.5 J/(kg K)
    >>> relative_humidity_partial_density(temperature = 273.15 + 21, partial_density = 0.01, specific_gas_constant = 461.5)
    ... 54.555...
    """

    e = partial_pressure(
        temperature=temperature,
        partial_density=partial_density,
        specific_gas_constant=specific_gas_constant,
    )

    # e_sat = saturation_vapour_pressure(temperature)
    # e_sat = saturation_vapour_pressure_Bolton(temperature)
    e_sat = saturation_vapour_pressure_murphy_koop(temperature)

    relhum = relative_humidity(saturation_vapour_pressure=e_sat, vapour_pressure=e)
    relhum.attrs.update(
        comment="Relative humidity calculated from the partial density of the water vapour",
    )

    return relhum


def relative_humidity_dewpoint(
    temperature: xr.DataArray, dew_point_temperature: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the relative humidity from the temperature and dew point temperature.
    The relative humidity is calculated using the saturation vapor pressure
    at the dew point temperature and the saturation vapor pressure at the
    temperature.

    Parameters
    ----------
    temperature : xr.DataArray
        The temperature in degrees Celsius.
    dew_point_temperature : xr.DataArray
        The dew point temperature in degrees Celsius.

    Returns
    -------
    xr.DataArray
        The relative humidity as a percentage.
    """

    e_dew = saturation_vapour_pressure_murphy_koop(dew_point_temperature)
    e_sat = saturation_vapour_pressure_murphy_koop(temperature)

    relhum = relative_humidity(
        saturation_vapour_pressure=e_sat,
        vapour_pressure=e_dew,
    )

    relhum.attrs.update(
        comment="Relative humidity calculated from the dew point temperature",
    )

    return relhum


def relative_humidity_from_tps(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    specific_humidity: xr.DataArray,
    simplified: bool = False,
) -> xr.DataArray:
    """
    Calculate the relative humidity from the temperature, pressure and specific
    humidity.

    Parameters
    ----------
    temperature : xr.DataArray
        The temperature in Kelvin.
    pressure : xr.DataArray
        The pressure in Pa.
    specific_humidity : xr.DataArray
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

    relhum = relative_humidity(
        saturation_vapour_pressure=es,
        vapour_pressure=e,
    )
    relhum.attrs.update(
        comment="Relative humidity calculated from temperature, pressure and specific humidity.",
    )
    return relhum


def potential_temperature_from_tp(
    air_temperature: xr.DataArray,
    pressure: xr.DataArray,
    pressure_reference: Union[
        float, np.ndarray, xr.DataArray
    ] = 100000,  # default value used for drop sondes dataset
    R_over_cp: float = 0.286,
):
    """
    Calculate the potential temperature from the air temperature and the pressure.

    Parameters
    ----------
    air_temperature : xr.DataArray or xr.DataArray
        The air temperature in Kelvin.
    pressure : xr.DataArray  or xr.DataArray
        The pressure in Pa.
    pressure_reference : float
        The reference pressure in Pa.
    R_over_cp : float, optional
        The ratio of the gas constant of air to the specific heat capacity at
        constant pressure. Default is 0.286.

    Returns
    -------
    np.ndarray
        The potential temperature in Kelvin.
    """
    theta = air_temperature * (pressure_reference / pressure) ** R_over_cp
    attrs = dict(
        name="potential_temperature",
        units="K",
        long_name="Potential temperature",
        description="Potential temperature calculated from air temperature and pressure.",
    )
    theta = __attrs_if_dataarray__(da=theta, attrs=attrs)
    return theta
