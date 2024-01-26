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


def test_consecutive_events_np_old():
    """Tests the consecutive_events_np function
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
    # Set up example array
    mask = np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )

    # --------------
    # Test 1
    # min_duration = 1
    expected_result_1 = mask

    # Axis = 1
    result_1a = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=1,
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    # Axis = 0
    result_1b = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=0,
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    np.testing.assert_array_equal(result_1b, expected_result_1)

    # --------------
    # Test 2
    # min_duration = 3
    # Axis = 1

    result_2 = consecutive_events_np(
        mask=mask,
        min_duration=3,
        axis=1,
    )
    expected_result_2 = np.array(
        [
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(result_2, expected_result_2)

    # --------------
    # Test 3
    # min_duration = 3
    # Axis = 0 which is the shorter one in the arrays

    result_3 = consecutive_events_np(
        mask=mask,
        min_duration=3,
        axis=0,
    )
    expected_result_3 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(result_3, expected_result_3)

    # Test 4
    # Same as Test 2 but with axis = 0 and tranpsosed mask
    result_4 = consecutive_events_np(mask=mask.T, min_duration=3, axis=0)
    np.testing.assert_array_equal(result_4, expected_result_2.T)

    # Test 5
    result_5 = consecutive_events_np(mask=mask, min_duration=0, axis=0)
    np.testing.assert_array_equal(result_5, np.zeros_like(mask))


@pytest.fixture
def mask():
    return np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )


@pytest.mark.parametrize(
    "min_duration, axis, expected",
    [
        # Test 1
        # min_duration = 1
        #  a)
        # axis = 1, which is the longer one in the arrays
        (
            1,
            1,
            np.array(
                [
                    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        #  b)
        # axis = 0, which is the shorter one in the arrays
        (
            1,
            0,
            np.array(
                [
                    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        # Test 2
        # min_duration = 3
        # axis = 1
        (
            3,
            1,
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        # Test 3
        # min_duration = 3
        # axis = 0
        (
            3,
            0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                ],
                dtype=bool,
            ),
        ),
        # Test 4
        # min_duration = 0
        # results should be zeros
        (0, 0, np.zeros_like(mask)),
        (0, 1, np.zeros_like(mask)),
    ],
)
def test_consecutive_events_np(mask, min_duration, axis, expected):
    print(mask, min_duration, axis, expected)
    result = consecutive_events_np(mask, min_duration=min_duration, axis=axis)
    np.testing.assert_array_equal(result, expected)


def test_consecutive_events_xr():
    """Tests the consecutive_events_xr function
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
    # Set up example array
    mask_np = np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    mask = xr.DataArray(
        mask_np,
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )

    # --------------
    # Test 1
    # Check for min_duration = 1

    expected_result_1 = xr.DataArray(
        mask_np,
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )

    # Dimension time
    result_1a = consecutive_events_xr(
        da_mask=mask,
        min_duration=1,
        axis="time",
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    xr.testing.assert_identical(result_1a, expected_result_1)

    # Dimension space
    result_1b = consecutive_events_xr(
        da_mask=mask,
        min_duration=1,
        axis="space",
    )
    np.testing.assert_array_equal(result_1b, expected_result_1)
    xr.testing.assert_identical(result_1b, expected_result_1)

    # --------------
    # Test 2
    # min_duration = 3
    # Axis = "space"

    result_2 = consecutive_events_xr(
        da_mask=mask,
        min_duration=3,
        axis="space",
    )
    expected_result_2 = xr.DataArray(
        np.array(
            [
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_2, expected_result_2)
    xr.testing.assert_identical(result_2, expected_result_2)

    # --------------
    # Test 3
    # min_duration = 3
    # Axis = time which is the shorter one in the arrays

    result_3 = consecutive_events_xr(
        da_mask=mask,
        min_duration=3,
        axis="time",
    )

    expected_result_3 = xr.DataArray(
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            ],
            dtype=bool,
        ),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_3, expected_result_3)
    xr.testing.assert_identical(result_3, expected_result_3)

    # Test 4
    # ----------------
    # min_duration = 0

    result_4a = consecutive_events_xr(
        da_mask=mask,
        min_duration=0,
        axis="time",
    )

    result_4b = consecutive_events_xr(
        da_mask=mask,
        min_duration=0,
        axis="space",
    )

    expected_result_4 = xr.DataArray(
        np.zeros_like(mask_np),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_4a, expected_result_4)
    np.testing.assert_array_equal(result_4b, expected_result_4)
    xr.testing.assert_identical(result_4a, expected_result_4)
    xr.testing.assert_identical(result_4b, expected_result_4)


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


def test_match_clouds_and_dropsondes():
    """
    Tests the match_clouds_and_dropsondes function. It uses a cloud dataset and
    a dropsonde dataset.

    ds_cloud:
        ```
        <xarray.Dataset>
        Dimensions:   (time: 1)
        Coordinates:
        * time      (time) datetime64[ns] 2020-01-02T06:00:00
        Data variables:
            cloud_id  (time) int64 1
            start     (time) datetime64[ns] 2020-01-01T03:00:00
            end       (time) datetime64[ns] 2020-01-01T09:00:00
        ```

    The dropsonde dataset has the following dates:
        ```
        <xarray.Dataset>
        Dimensions:  (time: 5)
        Coordinates:
        * time     (time) datetime64[ns] 2020-01-01 2020-01-02 ... 2020-01-05
        Data variables:
            temp     (time) int64 0 1 2 3 4
        ```

    The distance dataset has the following distances:
    - Temporal distances are by definition:
        30 h, 6h, -18h, -42h, -66h
    - Spatial distances are chosen:
        100 km, 100 km, 90 km, 120 km, 120 km. ()

    The distance dataset looks like this:
        ```
        <xarray.Dataset>
        Dimensions:                 (time_identified_clouds: 1, time_drop_sondes: 5)
        Coordinates:
        * time_drop_sondes        (time_drop_sondes) datetime64[ns] 2020-01-01 ... ...
        * time_identified_clouds  (time_identified_clouds) datetime64[ns] 2020-01-0...
        Data variables:
            spatial_distance        (time_identified_clouds, time_drop_sondes) int64 ...
            temporal_distance       (time_identified_clouds, time_drop_sondes) timedelta64[ns] ...
        ```

    The expected result is a dropsonde dataset with the following dates:
        ```
        <xarray.Dataset>
        Dimensions:  (time: 2)
        Coordinates:
        * time     (time) datetime64[ns] 2020-01-02 2020-01-03
        Data variables:
            temp     (time) int64 1 2
        ```
    """
    # Create a cloud dataset
    ds_cloud = xr.Dataset(
        {
            "cloud_id": (("time",), [1]),
            "start": (("time",), [pd.Timestamp("2020-01-01 3:00")]),
            "end": (("time",), [pd.Timestamp("2020-01-01 9:00")]),
        },
        coords={
            "time": [pd.Timestamp("2020-01-02 6:00")],
        },
    )

    # Create a dropsonde dataset
    ds_sonde = xr.Dataset(
        {
            "temp": (("time",), np.arange(5)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=5),
        },
    )

    # Create a distance dataset
    ds_distance = xr.Dataset(
        {
            "spatial_distance": (
                ("time_identified_clouds", "time_drop_sondes"),
                np.array([[100, 100, 90, 120, 120]], dtype="int"),
            ),
            "temporal_distance": (
                ("time_identified_clouds", "time_drop_sondes"),
                np.array([[30, 6, -18, -42, -66]], dtype="timedelta64[h]"),
            ),
        },
        coords={
            "time_drop_sondes": ds_sonde.time.data,
            "time_identified_clouds": ds_cloud.time.data,
        },
    )

    result = match_clouds_and_dropsondes(
        ds_cloud,
        ds_sonde,
        ds_distance,
        dim_in_dropsondes="time",
        index_ds_dropsonde="time_drop_sondes",
        index_ds_cloud="time_identified_clouds",
        max_temporal_distance=np.timedelta64(18, "h"),
        max_spatial_distance=100,
    )
    # the should results include the following dates:
    # - 2020-01-02 with temp = 1
    # - 2020-01-03 with temp = 2

    should = xr.Dataset(
        {
            "temp": (("time",), [1, 2]),
        },
        coords={
            "time": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
        },
    )
    assert isinstance(result, xr.Dataset)
    xr.testing.assert_identical(result, should)
