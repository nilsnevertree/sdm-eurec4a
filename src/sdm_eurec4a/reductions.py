from typing import Tuple, Union, Literal

import numpy as np
import scipy.interpolate
import pandas as pd
import xarray as xr

from dask import array as dask_array
from shapely.geometry import Point, Polygon


def polygon2mask(
    dobj: Union[xr.DataArray, xr.Dataset], pg: Polygon, lat_name: str = "lat", lon_name: str = "lon"
):
    """
    This funciton creates a mask for a given DataArray or DataSet based on a shapely
    Polygon or MultiPolygon. Polygon points are expected be (lon, lat) tuples. To fit
    the polygon to the dobj coords, "polygon_split_arbitrary" function is used. The dobj
    is expected to have lon values in [0E, 360E) coords and lat values in [90S, 90N]
    coords.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    pg: shapely Polygon or shapely MultiPolygon
        Polygon including the area wanted.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mask for given Polygon on dobj grid.
    """

    # handle prime meridian crossing Polygons and transform [180W, 180E) coords into [0E,360E) coords
    # pg = polygon_split_arbitrary(
    #     pg=pg,
    #     lat_max=90,
    #     lat_min=-90,
    #     lon_max=360,
    #     lon_min=0,
    # )

    # create the mask
    lon_2d, lat_2d = xr.broadcast(dobj[lon_name], dobj[lat_name])

    mask = xr.DataArray(
        np.reshape(
            [
                pg.contains(Point(_lon, _lat)) | pg.boundary.contains(Point(_lon, _lat))
                for _lon, _lat in zip(np.ravel(lon_2d, order="C"), np.ravel(lat_2d, order="C"))
            ],
            lon_2d.shape,
            order="C",
        ),
        dims=lon_2d.dims,
        coords=lon_2d.coords,
    )
    # transpose to ensure the same order of horizontal dims as the input object
    mask = mask.transpose(*[d for d in dobj.dims if d in [lon_name, lat_name]])

    return mask


def rectangle_spatial_mask(
    ds: xr.Dataset,
    area: Union[dict, list],
    lon_name: str = "lon",
    lat_name: str = "lat",
    include_boundary: bool = True,
) -> xr.DataArray:
    """
    Select a region from a xarray dataset based on a given area. The area can be defined
    as a dictionary with keys ['lon_min', 'lon_max', 'lat_min', 'lat_max'] or as a list
    of four values [lon_min, lon_max, lat_min, lat_max].

    The ``lon_name`` and ``lat_name`` parameters can be used to specify the names of the coords or variables where the
    longitude and latitude values are stored. By default, the function assumes that the longitude and latitude values
    are stored in variables named 'lon' and 'lat', respectively.

    The ``include_boundary`` parameter can be used to specify whether the boundary of the selected region should be
    included in the mask. By default, the boundary is included in the mask.

    The function returns a DataArray with the selected region marked as True. The DataArray has the same dimensions as
    longitude and latitude.
    Thus, if the latitude and longitude are not coordinates, the DataArray will have the same dimensions as the
    latitude and longitude variables.


    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to select from.
    area : Union(dict, list)
        Dictionary with keys ['lon_min', 'lon_max', 'lat_min', 'lat_max'] or
        List of four values [lon_min, lon_max, lat_min, lat_max].
    lon_name : str, optional
        Name of the longitude variable. The default is 'lon'.
    lat_name : str, optional
        Name of the latitude variable. The default is 'lat'.
    include_boundary : bool, optional
        Whether to include the boundary of the selected region in the mask. The default is True.

    Returns
    -------
    ds : xarray.DataArray
        DataArray with the selected region marked as True.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from sdm_eurec4a import reductions
    >>>
    >>> # Create a sample dataset
    >>> ds = xr.Dataset(
    ...     coords=dict(
    ...         lon=np.arange(0, 10),
    ...         lat=np.arange(-5, 5),
    ...     )
    ... )
    >>>
    >>> # Define the area of interest
    >>> # Include bounds and dict
    >>> area_dict = dict(lat_min=-5, lat_max=5, lon_min=0, lon_max=10)
    >>>
    >>> # Call the function
    >>> result_dict = reductions.rectangle_spatial_mask(
    ...     ds=ds,
    ...     area=area_dict,
    ...     include_boundary=True
    ... )
    """

    if isinstance(area, list):
        area = dict(lat_min=area[2], lat_max=area[3], lon_min=area[0], lon_max=area[1])

    if include_boundary:
        mask = (
            (ds[lon_name] >= area["lon_min"])
            & (ds[lon_name] <= area["lon_max"])
            & (ds[lat_name] >= area["lat_min"])
            & (ds[lat_name] <= area["lat_max"])
        )
    else:
        mask = (
            (ds[lon_name] > area["lon_min"])
            & (ds[lon_name] < area["lon_max"])
            & (ds[lat_name] > area["lat_min"])
            & (ds[lat_name] < area["lat_max"])
        )

    return mask


def latlon_dict_to_polygon(area):
    """
    Create a shapely polygon from a dictionary with lat lon values.

    Input:
        area: dict with keys ['lon_min', 'lon_max', 'lat_min', 'lat_max']
    Output:
        shapely.geometry.Polygon
        with x values as lon and y values as lat
    """
    return Polygon(
        [
            (area["lon_min"], area["lat_min"]),
            (area["lon_min"], area["lat_max"]),
            (area["lon_max"], area["lat_max"]),
            (area["lon_max"], area["lat_min"]),
        ]
    )


def x_y_flatten(da: xr.DataArray, axis: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Flatten a 2D data array along the specified axis.

    Note
    ----
    It returns two arrays: x and y.
    x contains the flattened values of the data array along the specified axis.
    y contains the values of ``axis`` corresponding to the values of y .

    Parameters
    ----------
    da : xr.DataArray
        The data array to be flattened.
        Only 2D data arrays are supported!
    axis : str
        The axis along which the data array is flattened.

    Returns
    -------
    np.ndarray
        The flattened data array along the specified axis.
    np.ndarray
        The values of ``axis`` corresponding to the values of y .

    Raises
    ------
    ValueError
        If the data array is > 2D.


    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr

    >>> da = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(2),
            "lat": np.arange(3),
            "lon": np.arange(4),
        },

    >>> x, y = x_y_flatten(da, "time")
    >>> print(x.shape)
    (24,)
    >>> print(y.shape)
    (24,)
    >>> print(x)
    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    >>> print(y)
    [0 0 0 0 1 1 1 1 2 2 2 2]
    """

    da = da.transpose(axis, ...)
    dims = da.dims
    da = da.stack(z=tuple(dims))
    return da[axis], da


def shape_dim_as_dataarray(da, output_dim: str):
    """
    Reshapes the dimension ``output_dim`` to the same shape as the given DataArray
    ``da``. Therefore the dimension ``output_dim`` is expanded to the same shape as the
    DataArray ``da``.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray including the data with dimension ``output_dim`` in it.
    output_coord : str
        The name of the dimension in the DataArray to be returned with the same shape and dims as ``da``.

    Returns
    -------
    xarray.DataArray
        DataArray containing values of ``output_dim`` and with same dims and shape as ``da``.

    Raises
    ------
    KeyError
        If the specified dimension is not in the DataArray.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr

    >>> da = xr.DataArray(
    ...     np.random.rand(4, 3, 5, 2),
    ...     dims=('x', 'y', 'z', 'time'),
    ...     coords = {
    ...         "x" : np.arange(0,4,1),
    ...         "y" : np.arange(0,3,1),
    ...         "z" : np.arange(0,5,1),
    ...         "time" : pd.date_range("2000-01-01", periods=2, freq="D")}
    ...     )

    >>> res = shape_dim_as_dataarray(da, 'time')

    >>> print(res)
    ... <xarray.DataArray 'time' (x: 4, y: 3, z: 5, time: 2)>
    ... array([[[['2000-01-01T00:00:00.000000000',
    ...      '2000-01-02T00:00:00.000000000'],
    ...  ...
    ...      ['2000-01-01T00:00:00.000000000',
    ...       '2000-01-02T00:00:00.000000000']]]], dtype='datetime64[ns]')
    ... Coordinates:
    ...   * x        (x) int64 0 1 2 3
    ...   * y        (y) int64 0 1 2
    ...   * z        (z) int64 0 1 2 3 4
    ...   * time     (time) datetime64[ns] 2000-01-01 2000-01-02
    """

    if output_dim not in da.dims:
        raise KeyError(f"The coordinate {output_dim} is not in the DataArray")

    axis_list = list()
    expand_dict = dict()
    for dim in da.dims:
        if dim == output_dim:
            pass
        else:
            expand_dict[dim] = da[dim]
            axis_list.append(da.dims.index(dim))

    return da[output_dim].expand_dims(dim=expand_dict, axis=axis_list)


def validate_datasets_same_attrs(datasets: list, skip_attrs: list = []) -> np.bool_:
    """
    Check if all datasets have the same attributes except for the ones in skip_attrs.

    Args:
        datasets (list): list of datasets
        skip_attrs (list): list of attributes to skip, default is empty list

    Returns:
        bool: True if all attributes are the same, False otherwise
    """
    attrs = []
    for ds in datasets:
        attrs.append(ds.attrs)
    attrs = pd.DataFrame(attrs)
    attrs = attrs.drop(columns=skip_attrs)
    nunique_attrs = attrs.nunique()
    return np.all(nunique_attrs == 1)


def mean_and_stderror_of_mean(
    data: xr.DataArray,
    dims: Union[Tuple[str], str],
    keep_attrs: bool = True,
    min_sample_size: int = 1,
    data_std: Union[xr.DataArray, None] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the standard error of the mean as given by the Central Limit Theorem.
    The formular for the std error of the mean is given by:
    .. math::
        SEM = \frac{std(data)}{\sqrt{N}}
    where :math:`N` is the number of samples in the data.

    For mulitple dimension, error propagation is used to calculate the standard error of the mean.
    The error propagation is given by:
    .. math::
        SEM = \sqrt{\frac{std(data)^2}{N} + \sum_{i=1}^{n} \frac{std(data_i)^2}{N^2}}

    The initial standard error of the data can be provided using ``data_std``.

    Parameters
    ----------
    data : xr.DataArray
        The input data array.
        Dimensions of the ``data`` array must contain at least the dimensions in the ``dims`` list.
    dims : list[str]
        List of dimensions along which the mean is calculated.
    keep_attrs : bool, optional
        Whether to keep the attributes of the data array. Default is True.
    min_sample_size : int, optional
        Minimum number of samples required to calculate the mean. Default is 1.
    data_std : xr.DataArray, optional
        Standard error of the data.
        Needs to be of the same shape as the ``data`` array.
        Default is None for no standard error of the data.

    Returns
    -------
    xr.DataArray
        Mean of the data array along the specified dimensions.
    xr.DataArray
        Standard error of the mean of the data array.
        As propagated error of the mean.


    Reference
    ---------
    https://en.wikipedia.org/wiki/Standard_error
    https://en.wikipedia.org/wiki/Central_limit_theorem
    """

    # make sure to have dims as a tuple
    if isinstance(dims, (list, tuple)):
        dims = dims
    elif isinstance(dims, str):
        dims = (dims,)
    else:
        raise TypeError("dims must be a list of strings")

    attrs = data.attrs.copy()
    name = data.name
    # calculate the first dimension in the list
    dim = dims[0]
    # mean
    m = data.mean(dim=dim)
    # standard error of the mean (SEM) for all valid points.
    # compute the sample size along the dimension
    sample_size = (~data.isnull()).sum(dim)
    sample_size = sample_size.where(sample_size >= min_sample_size, other=1)
    if data_std is None:
        s = data.std(dim=dim) / np.sqrt(sample_size)
    else:
        s = (data.std(dim=dim) ** 2 / sample_size + (data_std**2).sum(dim=dim) / sample_size**2) ** (0.5)

    if len(dims) > 1:
        for dim in dims[1:]:
            mm = m.mean(dim)
            sample_size = (~m.isnull()).sum(dim)
            sample_size = sample_size.where(sample_size >= min_sample_size, other=1)
            ss = (m.std(dim=dim) ** 2 / sample_size + (s**2).sum(dim=dim) / sample_size**2) ** (0.5)
            m = mm
            s = ss
    else:
        pass

    if keep_attrs == True:
        m.attrs.update(attrs)
        s.attrs.update(attrs)
        m.name = name
        s.name = name
    return m, s


def mean_and_stderror_of_mean_dataset(
    ds: xr.Dataset,
    dims: Union[Tuple[str], str],
    keep_attrs: bool = True,
    min_sample_size: int = 1,
    ds_std: Union[xr.Dataset, None] = None,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate the standard error of the mean as given by the Central Limit Theorem for a dataset.
    The formular for the std error of the mean is given by:
    .. math::
        SEM = \frac{std(data)}{\sqrt{N}}
    where :math:`N` is the number of samples in the data.

    This function is a wrapper around the ``mean_and_stderror_of_mean`` function and applies it to all data variables in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
        Dimensions of the ``data`` array must contain at least the dimensions in the ``dims`` list.
    dims : list[str]
        List of dimensions along which the mean is calculated.
    keep_attrs : bool, optional
        Whether to keep the attributes of the data variables. Default is True.
    min_sample_size : int, optional
        Minimum number of samples required to calculate the mean. Default is 1.
    ds_std : xr.Dataset, optional
        Standard error of the data.
        Needs to be similar to ``ds`` and contain the same vairables.
        Default is None for no standard error of the data.

    Returns
    -------
    xr.Dataset
        Mean of the data array along the specified dimensions.
    xr.Dataset
        Standard error of the mean of the data array.
        As propagated error of the mean.

    """

    # make sure to have dims as a tuple
    if isinstance(dims, (list, tuple)):
        dims = dims
    elif isinstance(dims, str):
        dims = (dims,)
    else:
        raise TypeError("dims must be a list of strings")

    ds_mean = xr.Dataset()
    ds_sem = xr.Dataset()

    for var in ds.data_vars:
        da = ds[var]

        # only apply on the data variables, which have the dimensions.
        if set(dims).issubset(set(da.dims)):
            if ds_std is not None:
                try:
                    data_std = ds_std[var]
                except KeyError:
                    raise KeyError(f"Variable {var} not found in the standard error dataset.")
            else:
                data_std = None

            # calcaulte the mean and sem for the variable
            m, s = mean_and_stderror_of_mean(
                data=ds[var],
                dims=dims,
                keep_attrs=keep_attrs,
                min_sample_size=min_sample_size,
                data_std=data_std,
            )
            ds_mean[var] = m
            ds_sem[var] = s
        else:
            ds_mean[var] = da
            ds_sem[var] = 0 * da

    return ds_mean, ds_sem


### Interpolation Dataset
def interpolate_omit_nan(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    method: Literal["linear", "cubic"] = "linear",
    kwargs: dict = {},
) -> np.ndarray:
    """
    Interpolate a 1-D function, omitting NaN values in the input arrays.
    Parameters
    ----------
    x : np.ndarray
        The x-coordinates at which to evaluate the interpolated values.
    xp : np.ndarray
        The x-coordinates of the data points, must be increasing.
    fp : np.ndarray
        The y-coordinates of the data points, same length as `xp`.
    method : str, optional
        The interpolation method to use. Supported methods are 'linear' and 'cubic'.
        Default is 'linear'.
    Returns
    -------
    np.ndarray
        The interpolated values, with NaNs where interpolation could not be performed.
    Notes
    -----
    - If there are fewer than 3 valid (non-NaN) points in `xp` and `fp`, the function
        will return an array of NaNs.
    - For 'cubic' interpolation, `scipy.interpolate.CubicSpline` is used.
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> xp = np.array([0, 1, 2, np.nan, 4])
    >>> fp = np.array([0, 1, 4, np.nan, 16])
    >>> interpolate_omit_nan(x, xp, fp)
    array([ 0.,  1.,  4., nan, 16.])
    """
    # Function implementation here

    # create mask for valid points
    mask = np.isfinite(xp) & np.isfinite(fp)

    # only run interpolation if there are at least 3 valid points
    if np.sum(mask) > 2:

        # apply mask to remove nans from the data
        xp = xp[mask]
        fp = fp[mask]

        # interpolate linear, using the np.interp function
        if method == "linear":
            result = np.interp(x=x, xp=xp, fp=fp, left=np.nan, right=np.nan, **kwargs)
        # interpolate cubic, using the scipy.interpolate.CubicSpline function
        elif method == "cubic":
            result = scipy.interpolate.interp1d(xp, fp, kind="cubic", **kwargs)(x)
    # otherwise, return array of NaNs
    else:
        result = np.full_like(x, np.nan)

    return result


# Pytest test routines
def test_interpolate_linear():
    x = np.array([0, 1, 2, 3, 4])
    xp = np.array([0, 1, 2, np.nan, 4])
    fp = np.array([0, 1, 4, np.nan, 16])
    expected = np.array([0.0, 1.0, 4.0, np.nan, 16.0])
    result = interpolate_omit_nan(x, xp, fp, method="linear")
    np.testing.assert_array_equal(result, expected)


def test_interpolate_cubic():
    x = np.array([0, 1, 2, 3, 4])
    xp = np.array([0, 1, 2, 3, 4])
    fp = np.array([0, 1, 8, 27, 64])
    expected = np.array([0.0, 1.0, 8.0, 27.0, 64.0])
    result = interpolate_omit_nan(x, xp, fp, method="cubic")
    np.testing.assert_array_equal(result, expected)


def test_interpolate_insufficient_points():
    x = np.array([0, 1, 2, 3, 4])
    xp = np.array([0, np.nan, np.nan, np.nan, 4])
    fp = np.array([0, np.nan, np.nan, np.nan, 16])
    expected = np.full_like(x, np.nan)
    result = interpolate_omit_nan(x, xp, fp, method="linear")
    np.testing.assert_array_equal(result, expected)


def interpolate_dataset(
    ds: xr.Dataset, mapped_dim: xr.DataArray, new_dim: xr.DataArray, old_dim_name: str
) -> xr.Dataset:
    """
    Iinterpolate a dataset along a new dimension, which has dependency on multiple coords.
    This function is an xarray wrapper for the `interpolate_omit_nan` function.

    Imagine a dataset with the dimensions [``bla``, ``flup``, ...] and you want to interpolate along the dimension ``bla``.
    You have a mapping, for instance a normalization array, which maps the old dimension ``bla`` to a new dimension.
    This mapping depends on dimension ``flup``.

    With this function, you can transform the dataset and replace the old dimension ``bla`` with the new dimension given in the mapping_dim.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to interpolate.
    mapped_dim : xr.DataArray
        The mapping of the old to the new dimension.
        So this DataArray should contain the values of the new dimension.
        It needs to have the old dimension as a coordinate.
    new_dim : xr.DataArray
        The new dimension to interpolate to.
        Its values need to correspond to the values of the `mapped_dim`.
        The DataArray should have a name.
    old_dim_name : str
        The name of the old dimension which shall be replaced.

    Returns
    -------
    xr.Dataset
        The interpolated dataset.


    """

    ds_interpolated = xr.Dataset()

    # perform the interpolation for all variables in the dataset
    for variable in ds.data_vars:
        dims = ds[variable].dims
        attrs = ds[variable].attrs.copy()
        data = ds[variable]

        # only apply on the data variables, which have the dimensions, from which the mapped dimension is a subset.
        # This means, if mapped dimension DataArray has dimensions [bla, flup] and bla is the old dimension,
        # the data variable should have at least the dimensions [bla, flup, ...]
        if set(mapped_dim.dims).issubset(set(dims)):

            result = xr.apply_ufunc(
                interpolate_omit_nan,
                new_dim,
                mapped_dim,
                data,
                input_core_dims=[[new_dim.name], [old_dim_name], [old_dim_name]],
                output_core_dims=[[new_dim.name]],
                vectorize=True,
                kwargs={"method": "linear"},
            )
            result.attrs.update(attrs)

            ds_interpolated[variable] = result
        else:
            ds_interpolated[variable] = data
    ds_interpolated[new_dim.name].attrs = new_dim.attrs.copy()
    return ds_interpolated
