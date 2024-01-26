import numpy as np
from sdm_eurec4a.calculations import great_circle_distance_np

def test_great_circle_distance_np():

    """
    Test the great_circle_distance_np function.
    
    Examples
    --------
    Results from https://www.nhc.noaa.gov/gccalc.shtml are also shown.
    Hamburg = 53.5488° N, 9.9872° E
    Berlin = 52.5200° N, 13.4050° E
    Sydney = 33.8688° S, 151.2093° E

    True distance of 
    1. Hamburg - Berlin: 255 km (rounded)
    2. Hamburg - Sydney: 16267 km (rounded)
    """

    should = [255, 16267] # km
    result = great_circle_distance_np(
            lon1 = 9.9872, 
            lat1 = 53.5488, 
            lon2 = np.array([13.4050, 151.2093]),
            lat2 = np.array([52.5200, -33.8688]),
            earth_radius=6371.0088
        )

    assert isinstance(result, np.ndarray)
    # allow relative tolerance of 0.5%
    np.testing.assert_allclose(result, should, rtol=5e-3)