# %%
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdm_eurec4a.reductions import (
    rectangle_spatial_mask,
    shape_dim_as_dataarray,
    validate_datasets_same_attrs,
    x_y_flatten,
)


def test_rectangle_spatial_mask():
    # Create a sample dataset
    ds = xr.Dataset(
        coords=dict(
            lon=("lon", np.arange(-10, 30, 10)),
            lat=("lat", np.arange(-10, 15, 5)),
            time=("time", pd.date_range("2000-01-01", periods=6)),
        ),
        data_vars=dict(
            temperature=(
                ["lon", "lat"],
                np.arange(4 * 5).reshape(4, 5),
            ),
        ),
        attrs=dict(
            description="Sample dataset",
        ),
    )

    should_bounds = np.array(
        [
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
        ]
    )

    should_nobounds = np.array(
        [
            [False, False, False, False, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )

    # Define the area of interest
    # Option 1 - include bounds and dict

    area_dict = dict(lat_min=-5, lat_max=5, lon_min=-10, lon_max=20)
    area_list = [
        -10,
        20,
        -5,
        5,
    ]

    # Call the select_region function
    result_dict_bounds = rectangle_spatial_mask(ds=ds, area=area_dict, include_boundary=True)
    result_dict_nobounds = rectangle_spatial_mask(ds=ds, area=area_dict, include_boundary=False)

    # Call the select_region function
    result_list_bounds = rectangle_spatial_mask(ds=ds, area=area_list, include_boundary=True)

    # Check if the result is a DataArray
    assert isinstance(result_dict_bounds, xr.DataArray)
    assert isinstance(result_list_bounds, xr.DataArray)

    # Check if the results from list and dict are the same
    assert result_dict_bounds.equals(result_list_bounds)

    # Only check result_dict_bounds as it equals result_list_bounds

    # Check if the result has the expected dimensions
    # These need to be the dimensions which lon and lat were defined on
    np.testing.assert_equal(sorted(result_dict_bounds.dims), sorted(ds.lon.dims + ds.lat.dims))

    # Check if the result are the expected values
    # Include boundary
    np.testing.assert_array_equal(result_dict_bounds.values, should_bounds)
    # Do not include boundary
    np.testing.assert_array_equal(result_dict_nobounds.values, should_nobounds)


# %%
def test_rectangle_spatial_mask_lat_lon_nodims():
    # Create a sample dataset
    ds = xr.Dataset(
        coords=dict(
            time=("time", pd.date_range("2000-01-01", periods=6)),
        ),
        data_vars=dict(
            temperature=(
                ["time"],
                np.arange(6),
            ),
            lon=(["time"], [-3, -2, -1, 0, 1, 2]),
            lat=(["time"], [2, 3, 4, 5, 6, 7]),
        ),
        attrs=dict(
            description="Sample dataset",
        ),
    )

    should_bounds = np.array(
        [False, True, True, True, False, False],
    )

    should_nobounds = np.array(
        [False, False, True, False, False, False],
    )

    # Define the area of interest
    # Option 1 - include bounds and dict

    area_dict = dict(lat_min=3, lat_max=6, lon_min=-3, lon_max=0)

    # Call the select_region function
    result_dict_bounds = rectangle_spatial_mask(ds=ds, area=area_dict, include_boundary=True)
    result_dict_nobounds = rectangle_spatial_mask(ds=ds, area=area_dict, include_boundary=False)

    # Check if the result is a DataArray
    assert isinstance(result_dict_bounds, xr.DataArray)
    # Only check result_dict_bounds as it equals result_list_bounds

    # Check if the result has the expected dimensions
    # These need to be the dimensions which lon and lat were defined on
    np.testing.assert_equal(
        sorted(result_dict_bounds.dims),
        sorted(np.unique(ds.lon.dims + ds.lat.dims)),
    )

    # Check if the result are the expected values
    # Include boundary
    np.testing.assert_array_equal(result_dict_bounds.values, should_bounds)
    # Do not include boundary
    np.testing.assert_array_equal(result_dict_nobounds.values, should_nobounds)


# %%
def test_x_y_flatten_DataArray():
    """Tests for the x_y_flatten function usage with a DataArray."""
    da = xr.DataArray(
        np.reshape(
            np.arange(2 * 4),
            newshape=(2, 4),
            order="C",
        ),
        dims=("dim_0", "dim_1"),
        coords={
            "dim_1": np.arange(4),
            "dim_0": np.arange(2),
        },
    )
    x, y = x_y_flatten(da, "dim_0")

    np.testing.assert_array_equal(x, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    np.testing.assert_array_equal(y, np.array([0, 1, 2, 3, 4, 5, 6, 7]))


def test_x_y_flatten_DataArray_3D():
    da = xr.DataArray(
        np.reshape(
            np.arange(24),
            newshape=(4, 3, 2),
        ),
        dims=("lon", "lat", "time"),
        coords={
            "time": np.arange(2),
            "lat": np.arange(3),
            "lon": np.arange(4),
        },
    )
    x, y = x_y_flatten(da, axis="time")

    np.testing.assert_array_equal(x, np.concatenate([np.zeros(3 * 4), np.ones(3 * 4)]))
    np.testing.assert_array_equal(y, np.concatenate([np.arange(0, 24, 2), np.arange(1, 24, 2)]))


def test_shape_dim_as_dataarray():
    da = xr.DataArray(
        np.random.rand(4, 3, 5, 2),
        dims=("x", "y", "z", "time"),
        coords={
            "x": np.arange(0, 4, 1),
            "y": np.arange(0, 3, 1),
            "z": np.arange(0, 5, 1),
            "time": pd.date_range("2000-01-01", periods=2, freq="D"),
        },
    )

    res = shape_dim_as_dataarray(da, "time")

    # Check that the output has the same dimensions and shape as the input
    assert da.dims == res.dims
    assert da.shape == res.shape
    # check that the choords and dims have the same values
    xr.testing.assert_equal(
        da.astype(float) * 0,
        res.astype(float) * 0,
    )

    # check that along all other dimension, the values of the output_dim are the same as in the input da
    for x in da["x"]:
        for y in da["y"]:
            for z in da["z"]:
                np.testing.assert_array_equal(res.isel(x=x, y=y, z=z), da["time"])

    # Check the KeyError
    with pytest.raises(KeyError) as e_info:
        shape_dim_as_dataarray(da, "time2")


empty_ds1 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 1,
        "random_string": "first random",
    },
)

empty_ds2 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 2,
        "random_string": "first random",
    },
)


empty_ds3 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 1,
        "random_string": "second random",
    },
)


# test to check using all combinations of the three empty datasets to validate if the attributes are the same or not
# The test function is parametrized with the three empty datasets
# and it usees the validate_datasets_same_attrs function to check if the attributes are the same or not
# \
def test_validate_datasets_same_attrs():
    # same dataset
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds1],
        )
        == True
    )
    # different number
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds2],
        )
        == False
    )
    # different string
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds3],
        )
        == False
    )
    # different number but skip string
    assert validate_datasets_same_attrs([empty_ds1, empty_ds2], skip_attrs=["random_string"]) == False
    # different number and skip number
    assert validate_datasets_same_attrs([empty_ds1, empty_ds2], skip_attrs=["random_number"]) == True
    # different string but skip number
    assert validate_datasets_same_attrs([empty_ds2, empty_ds3], skip_attrs=["random_number"]) == False
    # different string and number - skip both
    assert (
        validate_datasets_same_attrs(
            [empty_ds2, empty_ds3], skip_attrs=["random_number", "random_string"]
        )
        == True
    )
