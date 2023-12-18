# %%
import numpy as np
import pandas as pd
import xarray as xr

from sdm_eurec4a.reductions import rectangle_spatial_mask


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
