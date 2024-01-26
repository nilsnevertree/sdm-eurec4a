# %%

import numpy as np
import xarray as xr


def consecutive_events_xr(
    da_mask: xr.DataArray,
    min_duration: int = 1,
    axis: str = "time",
) -> xr.DataArray:
    """
    This function calculates the mask of consecutive events with a duration of
    at least ´´min_duration´´(default 1) It runs full on xarray and numpy.

    Input
    -----
    da_mask : xr.DataArray
        boolean array as xarray dataarray.
    min_duration : int, optional
        how many consecutive events the mask needs to be True, by default 1.
        Default results in a mask similar to da_mask.
    axis : str, optional
        Specifies along which axis the consecutive events shall be calculated, by default 'time'.

    Returns
    -------
    xr.DataArray
        mask array with True for each timestep that is part of a consecutive event

    Further Notes
    -------------

    Detailed explanation of calculation:
    (1 - True, 0 - False)
    Example
    >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
    >>> consecutive_events(da_mask, min_duration = 3) runs the following steps:
    First for loop: mask_temporary
    it will create the following temporary mask_index (start index always moved to the right.)
            [1,0,1,1,1,0,0,1,1,1]           [1,0,1,1,1,0,0,1,1,1]
            [0,1,1,1,0,0,1,1,1,1]             [0,1,1,1,0,0,1,1,1,1]
            [1,1,1,0,0,1,1,1,1,0]               [1,1,1,0,0,1,1,1,1,0]
            - - - - - - - - - -     # with & operator the following will be done:
    mask_temporary: [0,0,1,0,0,0,0,1,1,0]
    # if a heatwave occures, one can see that all values of the lists on the left side are 1 above each other

    but the mask_temporary is shorter ??
    -> for each value with 1 in mask_temporary the following 3 days (duration = 3) are part of the heatwave.

    Second for loop:
    mask_result     [0,0,0,0,0,0,0,0,0,0,0,0]       # mask_result starts with 0 ( need a array with the size to work on)
                     - - - - - - - - - - - -
    mask_temporary  [0,0,1,0,0,0,0,1,1,0]           # here the mask_temporary will be moved to the right for each itteration:
            ""        [0,0,1,0,0,0,0,1,1,0]         # now the + opertaor will be used (1+1=1, 1+0=1 , 0+0=0) :
            ""          [0,0,1,0,0,0,0,1,1,0]       # all 3 days after the 1 in mask_ temoprary will change to 1, as they are part of a heatwave.
                     - - - - - - - - - - - -
    mask_result:    [0,0,1,1,1,0,0,1,1,1,1,0]       # you should be convinced that this works :D
    da_mask : [1,0,1,1,1,0,0,1,1,1,1,0]
    """
    if da_mask[axis].ndim != 1:
        raise ValueError(f"axis must be one dimensional but is {da_mask[axis].ndim}")
    # make sure that the mask is boolean
    try:
        da_mask = da_mask.astype(bool)
    except Exception as e:
        raise ValueError(f"da_mask must be boolean. Error: {e}")
    # get the original axis order of the da_mask
    axis_order = da_mask.dims
    # reorder axis
    da_mask = da_mask.transpose(axis, ...)
    # get the length of the axis along which the consecutive events shall be calculated
    length = da_mask.shape[0]
    # make sure that the min_duration is smaller than the length of the axis
    if length < min_duration:
        raise ValueError(
            f"min_duration must be smaller than the length of the axis along which the consecutive events shall be calculated. min_duration: {min_duration}, length: {length}"
        )

    temporary = None
    for index in range(min_duration):
        # the selection always chooses a slice of the mask along time.
        # the selection always hat the same length but changing starting position.
        # the mask_orginal will be sliced along the time axis based on the selection index vauels e.g. [0,1,2,3,4,...., time_length - duration]
        current = da_mask.isel(
            {
                axis: np.arange(index, length - min_duration + index + 1),
            }
        ).data
        if temporary is None:
            temporary = current
        else:
            # the & operator is used to get True for each day where
            # afterwards at least for min_duration the mask_origianl is True (see Note)
            temporary = temporary & current

    # result should start with a False mask
    result = (da_mask * 0).astype(bool)
    for index in range(min_duration):
        # same slice as in 1. for loop
        selection = np.arange(index, length - min_duration + index + 1)
        # teh + operator will be used to identifiy each day ehich is part of the heat wave (see Note)
        result[selection] = result[selection] + temporary

    return xr.DataArray(result, dims=da_mask.dims, coords=da_mask.coords).transpose(*axis_order)


def consecutive_events_np(
    mask: np.ndarray,
    min_duration: int = 1,
    axis: int = 0,
) -> np.ndarray:
    """
    This function calculates the mask of consecutive events with a duration of
    at least ´´min_duration´´(default 1) It runs full on numpy.

    Note
    ----
    The provided array will be converted into a boolean array.
    This means that all values that are not 0 will be converted to True.
    Except for np.nan values or any other value that is not convertable to bool.

    Input
    -----
    mask : np.ndarray
        boolean array as numpy array.
    min_duration : int, optional
        how many consecutive events the mask needs to be True, by default 1.
        Default results in a mask similar to ``mask``.
        If min_duration = 0, the result will be a mask with False everywhere.
    axis : int, optional
        Specifies along which axis the consecutive events shall be calculated,
        by default 0.

    Returns
    -------
    np.ndarray
        mask array with True for each timestep that is part of a consecutive event

    Further Notes
    -------------

    Detailed explanation of calculation:
    (1 - True, 0 - False)
    Example
    >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
    >>> consecutive_events(da_mask, min_duration = 3) runs the following steps:
    First for loop: mask_temporary
    it will create the following temporary mask_index (start index always moved to the right.)
            [1,0,1,1,1,0,0,1,1,1]           [1,0,1,1,1,0,0,1,1,1]
            [0,1,1,1,0,0,1,1,1,1]             [0,1,1,1,0,0,1,1,1,1]
            [1,1,1,0,0,1,1,1,1,0]               [1,1,1,0,0,1,1,1,1,0]
            - - - - - - - - - -     # with & operator the following will be done:
    mask_temporary: [0,0,1,0,0,0,0,1,1,0]
    # if a heatwave occures, one can see that all values of the lists on the left side are 1 above each other

    but the mask_temporary is shorter ??
    -> for each value with 1 in mask_temporary the following 3 days (duration = 3) are part of the heatwave.

    Second for loop:
    mask_result     [0,0,0,0,0,0,0,0,0,0,0,0]       # mask_result starts with 0 ( need a array with the size to work on)
                     - - - - - - - - - - - -
    mask_temporary  [0,0,1,0,0,0,0,1,1,0]           # here the mask_temporary will be moved to the right for each itteration:
            ""        [0,0,1,0,0,0,0,1,1,0]         # now the + opertaor will be used (1+1=1, 1+0=1 , 0+0=0) :
            ""          [0,0,1,0,0,0,0,1,1,0]       # all 3 days after the 1 in mask_ temoprary will change to 1, as they are part of a heatwave.
                     - - - - - - - - - - - -
    mask_result:    [0,0,1,1,1,0,0,1,1,1,1,0]       # you should be convinced that this works :D
    da_mask : [1,0,1,1,1,0,0,1,1,1,1,0]
    """

    # make sure that the mask is boolean
    try:
        mask = mask.astype(bool)
    except Exception as e:
        raise ValueError(f"da_mask must be boolean. Error: {e}")
    # get the original axis order of the da_mask
    axis_order = np.arange(mask.ndim)
    # reorder axis
    mask = np.swapaxes(mask, axis, 0)
    # get the length of the axis along which the consecutive events shall be calculated
    length = mask.shape[0]
    # make sure that the min_duration is smaller than the length of the axis
    if length < min_duration:
        raise ValueError(
            f"min_duration must be smaller than the length of the axis along which the consecutive events shall be calculated. min_duration: {min_duration}, length: {length}"
        )

    temporary = None
    for index in range(min_duration):
        # the selection always chooses a slice of the mask along time.
        # the selection always hat the same length but changing starting position.
        # the mask_orginal will be sliced along the time axis based on the selection index vauels e.g. [0,1,2,3,4,...., time_length - duration]
        current = mask[index : length - min_duration + index + 1, ...]
        if temporary is None:
            temporary = current
        else:
            # the & operator is used to get True for each day where
            # afterwards at least for min_duration the mask_origianl is True (see Note)
            temporary = temporary & current

    # result should start with a False mask
    result = np.full_like(
        mask, fill_value=False
    )  # this just creates a mask with False everywhere (see Note)
    for index in range(min_duration):
        # same slice as in 1. for loop
        selection = np.arange(index, length - min_duration + index + 1)
        # teh + operator will be used to identifiy each day ehich is part of the heat wave (see Note)
        result[selection, ...] = result[selection, ...] + temporary

    # Make sure to change the view of mask back to original axis order
    mask = np.swapaxes(mask, axis, 0)
    result = np.swapaxes(result, axis, 0)
    return result


def select_individual_cloud_by_id(
    ds_clouds: xr.Dataset,
    chosen_id: int,
):
    """
    Select an individual cloud from the individual cloud dataset based on its ``cloud_id``
    The data for the individual cloud is extracted from the clouds dataset ``ds_clouds``.

    Parameters
    ----------
    ds_clouds : xr.Dataset
        Dataset containing data for all clouds from the individual cloud dataset.
        It is indexed by 'time'.
    chosen_id : int
        The id of the cloud to be selected.

    Returns
    -------
    xr.Dataset
        A subset of the individual cloud dataset that contains only the data for the specified cloud.
        The dataset contains the same variables as the input individual cloud dataset.
    """

    return ds_clouds.sel(time=ds_clouds["cloud_id"] == chosen_id)


def match_clouds_and_dropsondes(
    ds_cloud: xr.Dataset,
    ds_sonde: xr.Dataset,
    ds_distance: xr.Dataset,
    index_ds_cloud: str = "time_identified_clouds",
    index_ds_dropsonde: str = "time_drop_sondes",
    max_temporal_distance: np.timedelta64 = np.timedelta64(1, "h"),
    max_spatial_distance: float = 100,
    dask_compute: bool = True,
):
    """
    Returns the Subdataset of dropsondes based on their spatial and temporal
    distances to the individual cloud. The data for the individual cloud can be
    extracted from the individual cloud dataset. The dropsonde dataset is the
    Level 3 dropsonde dataset.

    Parameters
    ----------
    ds_cloud : xr.Dataset
        Dataset containing data for a single cloud from the individual cloud dataset.
        It is indexed by 'time'.
    ds_sonde : xr.Dataset
        Dataset containing dropsonde data. It is indexed by 'time_drop_sondes'.
    ds_distance : xr.Dataset
        Dataset containing distance data between clouds and dropsondes.
    index_ds_cloud : str, optional
        Index of the cloud dataset. Default is 'time'.
    index_ds_dropsonde : str, optional
        Index of the dropsonde dataset. Default is 'time_drop_sondes'.
    max_temporal_distance : np.timedelta64, optional
        Maximum allowed temporal distance between a cloud and a dropsonde for them to be considered a match.
        Default is 1 hour.
    max_spatial_distance : float, optional
        Units are in kilometers.
        Maximum allowed spatial distance (in the same units as ds_distance) between a cloud and a dropsonde for them to be considered a match.
        Default is 100.
    dask_compute : bool, optional
        If True, the output dataset is computed. Default is True.
        Dask will not be used if compute is False and then be lazy.

    Returns
    -------
    xr.Dataset
        A subset of the dropsonde dataset that matches with the cloud dataset based on the specified spatial and temporal distances.
        The dataset contains the same variables as the input dropsonde dataset.
    
    Example
    -------
    >>> ds_cloud = xr.Dataset(
    ...     {
    ...         "cloud_id": (("time",), [1, 2, 3]),
    ...         "start": (("time",), pd.date_range("2020-01-01", periods=3)),
    ...         "end": (("time",), pd.date_range("2020-01-03", periods=3)),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01", periods=3),
    ...     },
    ... )
    >>> ds_sonde = xr.Dataset(
    ...     {
    ...         "time_drop_sondes": (("time",), pd.date_range("2020-01-01", periods=5)),
    ...     },
    ...     coords={
    ...         "time_drop_sondes": pd.date_range("2020-01-01", periods=5),
    ...     },
    ... )
    >>> ds_distance = xr.Dataset(
    ...     {
    ...         "temporal_distance": (("time_drop_sondes", "time_identified_clouds"), np.random.rand(5, 3)),
    ...         "spatial_distance": (("time_drop_sondes", "time_identified_clouds"), np.random.rand(5, 3)),
    ...     },
    ...     coords={
    ...         "time_drop_sondes": pd.date_range("2020-01-01", periods=5),
    ...         "time_identified_clouds": pd.date_range("2020-01-01", periods=3),
    ...     },
    ... )
    >>> match_clouds_and_dropsondes(ds_cloud, ds_sonde, ds_distance)
    <xarray.Dataset>
    Dimensions:          (time_drop_sondes: 5)
    Coordinates:
        time_drop_sondes  (time_drop_sondes) datetime64[ns] 2020-01-01 ... 2020-01-05
    Data variables:
        *empty*
    """

    if ds_cloud["time"].shape != (1,):
        raise IndexError(
            f"The cloud dataset must contain only one cloud. Thus the shape of the time dimension must be (1,).\nBut it is {ds_cloud['time'].shape}"
        )

    # Extract the distance of a single cloud
    single_distances = ds_distance.sel({index_ds_cloud: ds_cloud["time"].data}).compute()

    # select the time of the dropsondes which are close to the cloud
    allowed_dropsonde_times = single_distances.where(
        # temporal distance
        (np.abs(single_distances["temporal_distance"]) <= max_temporal_distance)
        # spatial distance
        & (single_distances["spatial_distance"] <= max_spatial_distance),
        drop=True,
    )[index_ds_dropsonde].compute()

    # select the dropsondes which are close to the cloud
    if dask_compute is True:
        return ds_sonde.sel(time=allowed_dropsonde_times.data, drop=True).compute()
    else:
        return ds_sonde.sel(time=allowed_dropsonde_times, drop=True)


def match_clouds_and_cloudcomposite(
    ds_clouds: xr.DataArray,
    ds_cloudcomposite: xr.Dataset,
    dim: str = "time",
    var_name_start: str = "start",
    var_name_end: str = "end",
    dask_compute: bool = True,
):
    """
    Returns the subdataset of the cloud composite dataset which is part of
    multiple individual cloud. The selection is performed purely based on the
    start and end time of the individual cloud.

    Parameters
    ----------
    ds_cloud : xr.DataArray
        Dataset containing data for multiple clouds from the individual cloud dataset.
        It is indexed by 'time'.
    ds_cloudcomposite : xr.Dataset
        Dataset containing cloud composite data. It is indexed by ``dim``.
    dim : str, optional
        Name of the dimension along which the cloud composite dataset is indexed.
        Default is 'time'.
    var_name_start : str, optional
        Name of the variable in the cloud dataset that contains the start time of the individual cloud.
        Default is 'start'.
    var_name_end : str, optional
        Name of the variable in the cloud dataset that contains the end time of the individual cloud.
        Default is 'end'.
    dask_compute : bool, optional
        If True, the output dataset is computed. Default is True.
        Dask will not be used if compute is False and then be lazy.

    Returns
    -------
    xr.Dataset
        A subset of the cloud composite dataset that matches which is part of the individual cloud.
        The selection is performed purely based on the start and end time of the individual cloud.

    Example:
    --------
    >>> ds_cloudcomposite = xr.Dataset(
    ...     {
    ...         "cloud_composite": (("time", "lat", "lon"), np.random.rand(5, 5, 5)),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01", periods=5),
    ...         "lat": np.arange(5),
    ...         "lon": np.arange(5),
    ...     },
    ... )
    >>> ds_clouds = xr.Dataset(
    ...     {
    ...         "start": (("time",), pd.date_range("2020-01-01", periods=3)),
    ...         "end": (("time",), pd.date_range("2020-01-03", periods=3)),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01", periods=3),
    ...     },
    ... )
    >>> match_multiple_clouds_and_cloudcomposite(ds_clouds, ds_cloudcomposite)
    <xarray.Dataset>
    Dimensions:          (lat: 5, lon: 5, time: 3)
    Coordinates:
        * time             (time) datetime64[ns] 2020-01-01 2020-01-02 2020-01-03
        * lat              (lat) int64 0 1 2 3 4
        * lon              (lon) int64 0 1 2 3 4
    Data variables:
        cloud_composite  (time, lat, lon) float64 0.548 0.592 0.046 0.607 ... 0.236 0.604 0.959 0.22
    """

    # create a list of time arrays for each cloud
    time_list = []
    for start_time, end_time in zip(
        ds_clouds[var_name_start],
        ds_clouds[var_name_end],
    ):
        time_list.append(ds_cloudcomposite.sel({dim: slice(start_time, end_time)})[dim])

    # concatenate the list of time arrays and sort them
    time_array = xr.concat(time_list, dim=dim)
    time_array = time_array.sortby(dim)

    if dask_compute is True:
        return ds_cloudcomposite.sel({dim: time_array}).compute()
    else:
        return ds_cloudcomposite.sel({dim: time_array})
