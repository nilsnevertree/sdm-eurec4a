import numpy as np
import xarray as xr

from shapely.geometry import MultiPolygon, Point, Polygon


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
