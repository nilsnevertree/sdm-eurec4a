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
    Tests the match_clouds_and_cloudcomposite function For this a cloud composite and a
    cloud mask are created.

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


def ds_clouds():
    """Creates cloud dataset."""
    return xr.Dataset(
        {
            "cloud_id": (("time",), [0, 1]),
            "start": (("time",), pd.date_range("2020-01-01 3:00", periods=2, freq="D")),
            "end": (("time",), pd.date_range("2020-01-01 9:00", periods=2, freq="D")),
        },
        coords={
            "time": pd.date_range("2020-01-01 6:00", periods=2, freq="D"),
        },
    )


def ds_sonde():
    """Create a dropsonde dataset."""
    return xr.Dataset(
        {
            "temp": (("time",), np.arange(8)),
        },
        coords={
            "time": pd.date_range("2020-01-01 0:00", periods=8, freq="5h"),
        },
    )


def ds_distance():
    """Create a distance dataset."""
    # temporal_distance = ds_cloud.time.rename({"time": "time_identified_clouds"}) - ds_sonde.time.rename({"time": "time_drop_sondes"})
    # print(temporal_distance.data.astype("timedelta64[h]"))
    return xr.Dataset(
        {
            # spatial distance in km
            "spatial_distance": (
                ("time_identified_clouds", "time_drop_sondes"),
                np.array(
                    [
                        [1, 1, 3, 3, 5, 3, 1, 8],
                        [1, 1, 3, 3, 5, 3, 1, 8],
                    ],
                    dtype="int",
                ),
            ),
            # temporal distance in hours
            "temporal_distance": (
                ("time_identified_clouds", "time_drop_sondes"),
                np.array(
                    [[6, 1, -4, -9, -14, -19, -24, -29], [30, 25, 20, 15, 10, 5, 0, -5]],
                    dtype="timedelta64[h]",
                ).astype("timedelta64[ns]"),
            ),
        },
        coords={
            "time_drop_sondes": pd.date_range("2020-01-01 0:00", periods=8, freq="5h"),
            "time_identified_clouds": pd.date_range("2020-01-01 6:00", periods=2, freq="D"),
        },
    )


def selected_dropsondes_5h_3km():
    """Should result for max_temporal_distance = 5 hours max_spatial_distance = 3 km."""
    return xr.Dataset(
        {
            "temp": (("time",), [1, 2, 5, 6]),
        },
        coords={
            "time": [
                pd.Timestamp("2020-01-01 05:00"),
                pd.Timestamp("2020-01-01 10:00"),
                pd.Timestamp("2020-01-02 01:00"),
                pd.Timestamp("2020-01-02 06:00"),
            ],
        },
    )


def selected_dropsondes_1h_10km():
    """Should result for max_temporal_distance = 5 hours max_spatial_distance = 3 km."""
    return xr.Dataset(
        {
            "temp": (("time",), [1, 6]),
        },
        coords={
            "time": [
                pd.Timestamp("2020-01-01 05:00"),
                pd.Timestamp("2020-01-02 06:00"),
            ],
        },
    )


def selected_dropsondes_0h_0km():
    """Should result for max_temporal_distance = 0 hours max_spatial_distance = 0 km."""
    return xr.Dataset(
        {
            "temp": (("time",), []),
        },
        coords={
            "time": np.array([], dtype="datetime64[ns]"),
        },
    )


@pytest.mark.parametrize(
    "max_temporal_distance, max_spatial_distance, expected",
    [
        (5, 3, selected_dropsondes_5h_3km()),
        (1, 10, selected_dropsondes_1h_10km()),
        (0, 0, selected_dropsondes_0h_0km()),
    ],
)
def test_match_clouds_and_dropsondes(max_temporal_distance, max_spatial_distance, expected):
    """
    Tests the match_clouds_and_dropsondes function. It uses a cloud dataset and a
    dropsonde dataset.

    Example visualisation
        >>> # Example setup
        ... # For a max dt = 5 hours
        ... # For a max dh = 3 km
        ... # The datasets below can be summarized visually as follows:
        ... # ->    2020-01-01      <-|-> 2020-01-02
        ... # ______________________________________
        ... # 0    5    10   15   20  |1    6    11   # Time in hours
        ... # ______________________________________
        ... # ---S--M--E--------------|--S--M--E----  # Cloud start, middle, end time
        ... # D----D----D----D----D---|D----D----D--  # Dropsonde
        ... # 1----1----3----3----5---|3----1----8--  # Distance dropsonde to clouds in km (same for both clouds)
        ... # ______________________________________
        ... # F----T----T----F----F---|T----T----F--  # Dropsondes close to cloud T for true F for false
        ... # F----T----T----F----F---|F----F----F--  # Dropsondes close to cloud 0
        ... # F----T----T----F----F---|T----T----F--  # Dropsondes close to cloud 1
    """
    max_temporal_distance = np.timedelta64(max_temporal_distance, "h")
    result = match_clouds_and_dropsondes(
        ds_clouds=ds_clouds(),
        ds_sonde=ds_sonde(),
        ds_distance=ds_distance(),
        dim_in_dropsondes="time",
        dim_in_clouds="time",
        index_ds_dropsonde="time_drop_sondes",
        index_ds_clouds="time_identified_clouds",
        max_temporal_distance=max_temporal_distance,
        max_spatial_distance=max_spatial_distance,
        dask_compute=False,
    )
    assert isinstance(result, xr.Dataset)
    xr.testing.assert_equal(result, expected)


def test_match_clouds_and_dropsondes_fails():
    """Checks for correct fails with ValueError is not all values from ds_cloud are in
    ds_distance."""

    # Test 1:
    # Fails if values from ds_clouds arent in ds_distance

    ds_distance_subset = ds_distance().isel({"time_identified_clouds": 0})
    ds_clouds_subset = ds_clouds().isel({"time": 0})
    ds_sonde_subset = ds_sonde().isel({"time": 0})
    with pytest.raises(ValueError):
        match_clouds_and_dropsondes(
            ds_clouds=ds_clouds(),
            ds_sonde=ds_sonde(),
            ds_distance=ds_distance_subset,
            dim_in_dropsondes="time",
            dim_in_clouds="time",
            index_ds_dropsonde="time_drop_sondes",
            index_ds_clouds="time_identified_clouds",
            max_temporal_distance=np.timedelta64(5, "h"),
            max_spatial_distance=3,
        )
    # Test 2:
    # Does not fail if values from ds_clouds arent in ds_distance
    match_clouds_and_dropsondes(
        ds_clouds=ds_clouds_subset,
        ds_sonde=ds_sonde(),
        ds_distance=ds_distance(),
        dim_in_dropsondes="time",
        dim_in_clouds="time",
        index_ds_dropsonde="time_drop_sondes",
        index_ds_clouds="time_identified_clouds",
        max_temporal_distance=np.timedelta64(5, "h"),
        max_spatial_distance=3,
    )
    # Test 3:
    # Fails if values from ds_sonde arent in ds_distance and vice versa
    with pytest.raises(ValueError):
        match_clouds_and_dropsondes(
            ds_clouds=ds_clouds(),
            ds_sonde=ds_sonde(),
            ds_distance=ds_distance_subset,
            dim_in_dropsondes="time",
            dim_in_clouds="time",
            index_ds_dropsonde="time_drop_sondes",
            index_ds_clouds="time_identified_clouds",
            max_temporal_distance=np.timedelta64(5, "h"),
            max_spatial_distance=3,
        )
    # Test 4:
    with pytest.raises(ValueError):
        match_clouds_and_dropsondes(
            ds_clouds=ds_clouds(),
            ds_sonde=ds_sonde_subset,
            ds_distance=ds_distance(),
            dim_in_dropsondes="time",
            dim_in_clouds="time",
            index_ds_dropsonde="time_drop_sondes",
            index_ds_clouds="time_identified_clouds",
            max_temporal_distance=np.timedelta64(5, "h"),
            max_spatial_distance=3,
        )
