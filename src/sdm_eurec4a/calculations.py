import numpy as np


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
    >>> haversine_np(30, 20, 40, -10)
    3339.191017
    >>> haversine_np(np.array([30, 20]), np.array([40, -10]), np.array([40, -10]), np.array([30, 20]))
    array([3339.191017, 3339.191017])
    >>> haversine_np(np.array([30, 20]), np.array([40, -10]), np.array([40, -10]), np.array([30, 20]), earth_radius=1)
    array([0.52359878, 0.52359878])
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    res = earth_radius * c
    return res
