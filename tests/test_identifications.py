import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdm_eurec4a.identifications import (
    consecutive_events_np,
    consecutive_events_xr,
    match_clouds_and_cloudcomposite,
    match_clouds_and_dropsondes,
)

def mask_np():
    return np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )


def da_mask():
    return xr.DataArray(
        mask_np(),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )


def expected_result_1():
    return mask_np()


def da_expected_result_1():
    return xr.DataArray(
        expected_result_1(),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )


def expected_result_2():
    return np.array(
        [
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )


def da_expected_result_2():
    return xr.DataArray(
        expected_result_2(),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )


def expected_result_3():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ],
        dtype=bool,
    )


def da_expected_result_3():
    return xr.DataArray(
        expected_result_3(),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )


def expected_result_4():
    return np.zeros_like(mask_np())


def da_expected_result_4():
    return xr.DataArray(
        expected_result_4(),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )


@pytest.fixture
def mask():
    return mask_np()

@pytest.fixture
def mask_fail():
    return np.ones((12, 3), dtype=bool)



@pytest.mark.parametrize(
    "min_duration, axis, expected",
    [
        (1, 1, expected_result_1()),
        (1, 0, expected_result_1()),
        (3, 1, expected_result_2()),
        (3, 0, expected_result_3()),
        (0, 0, expected_result_4()),
        (0, 1, expected_result_4()),
    ],
)
def test_consecutive_events_np(mask, min_duration, axis, expected):
    """
    It handles the following cases:
    1. min_duration = 1
    2. min_duration = 3, axis = 1
    3. min_duration = 3, axis = 0
    4. min_duration = 3, axis = 0, mask transposed
    5. min_duration = 0

    The original mask has the following shape: (3,12)
    has cosecutive events of length 3 in
    - the middle of the array
    - both ends of the array
    """
    result = consecutive_events_np(mask, min_duration=min_duration, axis=axis)
    np.testing.assert_array_equal(result, expected)


def test_consecutive_events_np_boolfail():
    """Checks for correct fails with ValueError."""
    # Test 1
    # not convertiable to bool
    mask_convertable = np.array([np.nan, 0, 3])
    consecutive_events_np(mask_convertable, min_duration=1, axis=0)
    mask_non_convertable = np.array([np.nan, 0, "something"])
    with pytest.raises(ValueError):
        consecutive_events_np(mask_non_convertable, min_duration=1, axis=0)



@pytest.mark.parametrize(
    "min_duration, axis, fails",
    [
        (12, 0, False),
        (13, 0, True),
        (3, 1, False),
        (12, 1, True),
        (13, 1, True),
    ],
)
def test_consecutive_events_np_exceed_max_duration(mask_fail, min_duration, axis, fails):
    """Checks for correct fails with ValueError."""

    # Test 2
    if fails:
        with pytest.raises(ValueError):
            consecutive_events_np(mask_fail, min_duration=min_duration, axis=axis)
    else:
        consecutive_events_np(mask_fail, min_duration=min_duration, axis=axis)


@pytest.mark.parametrize(
    "min_duration, axis, expected",
    [
        (1, "time", da_expected_result_1()),
        (1, "space", da_expected_result_1()),
        (3, "space", da_expected_result_2()),
        (3, "time", da_expected_result_3()),
        (0, "space", da_expected_result_4()),
        (0, "time", da_expected_result_4()),
    ],
)
def test_consecutive_events_xr(min_duration, axis, expected, ds_mask=da_mask()):
    """
    Tests the consecutive_events_xr functiom
    Tests the consecutive_events_xr function
    It handles the following cases:
    1. min_duration = 1
    2. min_duration = 3, axis = "space"
    3. min_duration = 3, axis = "time"
    4. min_duration = 0

    The original mask has the following shape: (3,12)
    has cosecutive events of length 3 in
    - the middle of the array
    - both ends of the array
    """
    result = consecutive_events_xr(da_mask=ds_mask, min_duration=min_duration, axis=axis)
    xr.testing.assert_identical(result, expected)


def test_match_clouds_and_cloudcomposite():
    """
    Tests the match_clouds_and_cloudcomposite function For this a cloud
    composite and a cloud mask are created.

    The cloud composite dataset has the following dates:
    - 2020-01-01 03:00:00
    - 2020-01-02 03:00:00
    - 2020-01-03 03:00:00
    - 2020-01-04 03:00:00
    - 2020-01-05 03:00:00

    The cloud mask dataset allows the following slices:
    - 2020-01-01 00:00:00 to 2020-01-01 12:00:00
    - 2020-01-02 00:00:00 to 2020-01-02 12:00:00
    - 2020-01-03 00:00:00 to 2020-01-03 12:00:00

    The expected result is a cloud composite dataset with the following dates:
    - 2020-01-01 03:00:00
    - 2020-01-02 03:00:00
    - 2020-01-03 03:00:00
    """

    # Create a cloud composite
    ds_cloudcomposite = xr.Dataset(
        {
            "cloud_composite": (("time", "lat", "lon"), np.arange(5 * 5 * 5).reshape(5, 5, 5)),
        },
        coords={
            "time": pd.date_range("2020-01-01 3:00", periods=5),
            "lat": np.arange(5),
            "lon": np.arange(5),
        },
    )

    # Create a cloud mask
    #  here the start and end data only allow the cloud composite to be matched to the first three time steps
    ds_clouds = xr.Dataset(
        {
            "start": (("time",), pd.date_range("2020-01-01 0:00", periods=3)),
            "end": (("time",), pd.date_range("2020-01-01 12:00", periods=3)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=3),
        },
    )

    expected_result = ds_cloudcomposite.sel(
        time=[
            "2020-01-01 03:00:00",
            "2020-01-02 03:00:00",
            "2020-01-03 03:00:00",
        ]
    )

    result = match_clouds_and_cloudcomposite(
        ds_clouds=ds_clouds,
        ds_cloudcomposite=ds_cloudcomposite,
        dim="time",
    )
    assert isinstance(result, xr.Dataset)
    assert "cloud_composite" in result.data_vars
    xr.testing.assert_identical(result, expected_result)
