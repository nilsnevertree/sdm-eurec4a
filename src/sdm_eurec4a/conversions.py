import numpy as np
import xarray as xr


def psd2msd(
    ds: xr.Dataset,
    psd_dim: str = "particle_size_distribution",
    diameter_dim: str = "diameter",
    diameter_scale: float = 1e-6,
    rho_water: float = 1000,
) -> xr.DataArray:
    """
    Converts the particle size distribution (PSD) to the mass size distribution
    (MSD). this follows the formular.

    .. math::
        MSD = rho_{water} \\cdot PSD \\cdot \\frac{4}{3} \\pi \\cdot (d/2 \cdot scale)^2

    Parameters
    ----------
    ds : xarray.Dataset
        input dataset
    psd_dim : str, optional
        name of the PSD dimension, by default "particle_size_distribution"
        assumed to be in [#/L]
    diameter_dim : str, optional
        name of the diameter dimension, by default "diameter"
    diameter_scale : float, optional
        scale factor of the diameter, by default micrometer 1e-6
    rho_water : float, optional
        density of water, by default 1000

    Returns
    -------
    xarray.DataArray
        dataset with MSD
        MSD in [kg m^-3]
    """

    msd = ds[psd_dim] * rho_water * 4 / 3 * np.pi * (ds[diameter_dim] / 2 * diameter_scale) ** 3
    return msd


def msd2lwc(
    ds: xr.Dataset,
    msd_dim: str = "mass_size_distribution",
    diameter_dim: str = "diameter",
    diameter_scale: float = 1e-6,
    rho_water: float = 1000,
) -> xr.DataArray:
    """
    Converts the mass size distribution (MSD) to the liquid water content
    (LWC). this follows the formular.

    .. math::
        LWC = \\sum{MSD}

    Parameters
    ----------
    ds : xarray.Dataset
        input dataset
    msd_dim : str, optional
        name of the MSD dimension, by default "mass_size_distribution"
        assumed to be in [kg m^-3]
    diameter_dim : str, optional
        name of the diameter dimension, by default "diameter"
    diameter_scale : float, optional
        scale factor of the diameter, by default micrometer 1e-6
    rho_water : float, optional
        density of water, by default 1000

    Returns
    -------
    xarray.DataArray
        dataset with LWC
        LWC in [kg m^-3]
    """

    lwc = ds[msd_dim].sum(dim=diameter_dim)

    return lwc


def psd2lwc(
    ds: xr.Dataset,
    psd_dim: str = "particle_size_distribution",
    diameter_dim: str = "diameter",
    diameter_scale: float = 1e-6,
    rho_water: float = 1000,
) -> xr.DataArray:
    """
    Converts the particle size distribution (PSD) to the liquid water content
    (LWC). this follows the formular.

    .. math::
        To be added

    Parameters
    ----------
    ds : xarray.Dataset
        input dataset
    psd_dim : str, optional
        name of the PSD dimension, by default "particle_size_distribution"
        assumed to be in [#/L]
    diameter_dim : str, optional
        name of the diameter dimension, by default "diameter"
    diameter_scale : float, optional
        scale factor of the diameter, by default micrometer 1e-6
    rho_water : float, optional
        density of water, by default 1000

    Returns
    -------
    xarray.DataArray
        dataset with LWC
        LWC in [kg m^-3]
    """

    msd = psd2msd(
        ds=ds,
        psd_dim=psd_dim,
        diameter_dim=diameter_dim,
        diameter_scale=diameter_scale,
        rho_water=rho_water,
    )
    ds_msd = xr.Dataset(data_vars=dict(mass_size_distribution=msd))
    lwc = msd2lwc(
        ds=ds_msd,
        msd_dim="mass_size_distribution",
        diameter_dim=diameter_dim,
        diameter_scale=diameter_scale,
        rho_water=rho_water,
    )

    return lwc


msd = psd2msd(
    ds=cloud_composite,
    psd_dim="particle_size_distribution",
    diameter_dim="diameter",
    diameter_scale=1e-6,
    rho_water=1000,
)

lwc = psd2lwc(
    ds=cloud_composite,
    psd_dim="particle_size_distribution",
    diameter_dim="diameter",
    diameter_scale=1e-6,
    rho_water=1000,
)
