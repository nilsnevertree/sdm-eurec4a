from typing import Union

import dask
from dask import array as dask_array
import numpy as np
import xarray as xr

from shapely.geometry import Point, Polygon


def polygon2mask(lon, lat, pg, lat_name="lat", lon_name="lon"):
    """
    This funciton creates a mask for a given DataArray or DataSet based on a
    shapely Polygon or MultiPolygon. Polygon points are expected be (lon, lat)
    tuples. To fit the polygon to the dobj coords, "polygon_split_arbitrary"
    function is used. The dobj is expected to have lon values in [0E, 360E)
    coords and lat values in [90S, 90N] coords.

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
    lon_2d, lat_2d = xr.broadcast(lon, lat)

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
    Select a region from a xarray dataset based on a given area. The area can
    be defined as a dictionary with keys ['lon_min', 'lon_max', 'lat_min',
    'lat_max'] or as a list of four values [lon_min, lon_max, lat_min,
    lat_max].

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


def x_y_flatten(da: xr.DataArray, axis: str):
    """
    Flatten a 2D data array along the specified axis. The data array is
    flattened in Fortran order after the DataArray is transposed properly.

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

    if da.ndim > 2:
        raise ValueError(f"The data array must have max. 2 dimensions but has {da.ndim}.")

    da = da.transpose(axis, ...)
    y = da.data
    if isinstance(y, dask_array.core.Array):
        print("Loading the dask array into memory.")
        y = y.compute()

    x = da[axis].data
    x = np.tile(x, da.shape[-1])
    y = y.flatten(order="F")  # very important
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    return x, y


def shape_dim_as_dataarray(da, output_dim: str):
    """
    Reshapes the dimension ``output_dim`` to the same shape as the given DataArray ``da``.
    Therefore the dimension ``output_dim`` is expanded to the same shape as the DataArray ``da``.

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
