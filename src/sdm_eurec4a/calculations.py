import warnings

from typing import Tuple

import numpy as np
import xarray as xr

from sdm_eurec4a.identifications import match_clouds_and_cloudcomposite


def great_circle_distance_np(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    earth_radius: float = 6371.0088,
) -> np.ndarray:
    """
    This function calculates the great circle distance between two points on
    the earth. Both latitude and longitude shall be given in decimal degrees.
    It gives and estimate of the true distance between two points on the earth.
    Thus, it is not exact, but it is fast. Better results can be achieved with
    geopy.distance.distance.

    Notes
    -----
        - All latitude and longitude values must be of equal shape.
        - The error for long distances can be up to 0.5%. https://en.wikipedia.org/wiki/Great-circle_distance
        - Earth radius is chosen as mean radius from WGS84 as 6371.0088 km. https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84


    References
    ----------
    Forumla given by: https://en.wikipedia.org/wiki/Haversine_formula and https://en.wikipedia.org/wiki/Great-circle_distance
    Implementation taken from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas/29546836#29546836

    Parameters
    ----------
    lon1 : np.ndarray
        Longitude of the first point.
    lat1 : np.ndarray
        Latitude of the first point.
    lon2 : np.ndarray
        Longitude of the second point.
    lat2 : np.ndarray
        Latitude of the second point.
    earth_radius : float, optional
        Radius of the earth in km, by default 6371.0088 as the mean radius of the earth given by WGS84.

    Returns
    -------
    np.ndarray
        Distance between the two points in km.

    Examples
    --------
    Results from https://www.nhc.noaa.gov/gccalc.shtml are also shown.
    Hamburg = 53.5488° N, 9.9872° E
    Berlin = 52.5200° N, 13.4050° E
    Sydney = 33.8688° S, 151.2093° E

    >>> great_circle_distance_np(9.9872, 53.5488, 13.4050, 52.5200)
    255.52968037566168  # following link above gives 255 km (rounded) -> error of ca. 0.2%
    >>> great_circle_distance_np(9.9872, 53.5488, 151.2093, -33.8688)
    16278.138412904867   # following link above gives 16267 km (rounded) -> error of ca. 0.07%
    >>> great_circle_distance_np(9.9872, 53.5488, 151.2093, -33.8688, earth_radius=1)
    2.555033107614742
    >>> great_circle_distance_np(
            lon1 = 9.9872,
            lat1 = 53.5488,
            lon2 = np.array([13.4050, 151.2093]),
            lat2 = np.array([52.5200, -33.8688]),
        )
    array([  255.52968038, 16278.1384129 ])
    """
    # Convert 0 to 360 degrees latitudes to -180 to 180 degrees
    lat1 = np.where(lat1 > 180, lat1 - 360, lat1)
    lat2 = np.where(lat2 > 180, lat2 - 360, lat2)

    # Convert all values to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Calculate the differences between the latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Calculate the great circle distance
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    res = earth_radius * c
    return res


def horizontal_extend(
    da: xr.Dataset,
    lon_name: str = "lon",
    lat_name: str = "lat",
):
    """
    Calculate the horizontal extent of a Dataset in Meter [m]. It uses the
    great_circle_distance_np function but returns the horizontal extent in m.

    d_lon = da[lon_name].max() - da[lon_name].min()
    d_lat = da[lat_name].max() - da[lat_name].min()

    The horizontal extent is then the distance from the value a virtual point at (0,0) to a Point at (d_lat, d_lon)

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the cloud information.
    lon_name : str, optional
        Name of the longitude coordinate, by default "lon"
    lat_name : str, optional
        Name of the latitude coordinate, by default "lat"

    Returns
    -------
    xr.DataArray
        The spatial extent of the cloud in m.
    """
    d_lon = da[lon_name].max() - da[lon_name].min()
    d_lat = da[lat_name].max() - da[lat_name].min()
    spatial_extend = 1e3 * great_circle_distance_np(d_lat, d_lon, 0, 0)
    spatial_extend.name = "spatial_extend"
    spatial_extend.attrs.update(
        {
            "long_name": "spatial extent of cloud",
            "units": "km",
            "description": "The spatial extent of the cloud in km. Calculated as the great circle distance based on the minimum and maximum of both latitude and longitude.",
        }
    )
    return spatial_extend


def vertical_extend(
    ds: xr.Dataset,
    alt_name: str = "alt",
) -> xr.DataArray:
    """
    Calculate the vertical extent of a Dataset in the same unit as the altitude
    is given. The output units are the same as the input units.

    Notes
    -----
    The vertical extent is calculated as the difference between the maximum and minimum altitude.
    The known unit keys in the altitude attributes are ["units", "Units", "unit", "Unit"].

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the cloud information.
    alt_name : str, optional
        Name of the altitude coordinate, by default "alt"

    Returns
    -------
    xr.DataArray
        The vertical extent of the cloud in the same unit as ``ds[alt_name]``.
    """

    alt_attrs = ds[alt_name].attrs

    known_unit_keys = ["units", "Units", "unit", "Unit"]
    found = False
    for key in known_unit_keys:
        if key in alt_attrs:
            alt_units = alt_attrs[key]
            found = True
            break
    if not found:
        warnings.warn("No units found for altitude.")
        alt_units = "None"

    vertical_extend = ds[alt_name].max() - ds[alt_name].min()
    vertical_extend.name = "vertical_extend"
    vertical_extend.attrs.update(
        {
            "long_name": "vertical extent of cloud",
            "units": f"{alt_units}",
            "description": f"The vertical extent of the cloud in {alt_units}. Calculated as the difference between the maximum and minimum altitude.",
        }
    )
    return vertical_extend
