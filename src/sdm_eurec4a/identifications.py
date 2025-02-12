import numpy as np
import xarray as xr


def consecutive_events_xr(
    da_mask: xr.DataArray,
    min_duration: int = 1,
    axis: str = "time",
) -> xr.DataArray:
    """
    This function calculates the mask of consecutive events with a duration of at least
    ´´min_duration´´(default 1) It runs full on xarray and numpy.

    Parameters
    ----------
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


    Notes
    --------------------------
    (1 - True, 0 - False)

    Example setup
        >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
        >>> consecutive_events(da_mask, min_duration = 3)

    First for loop: mask_temporary
        >>> # It will create the following temporary mask_index (start index always moved to the right.)
        >>> [1,0,1,1,1,0,0,1,1,1]    # [1,0,1,1,1,0,0,1,1,1]
        >>> [0,1,1,1,0,0,1,1,1,1]    #   [0,1,1,1,0,0,1,1,1,1]
        >>> [1,1,1,0,0,1,1,1,1,0]    #     [1,1,1,0,0,1,1,1,1,0]
        >>>  - - - - - - - - - -     # with & operator the following will be done:
        >>> [0,0,1,0,0,0,0,1,1,0]    # <- mask_temporary:

    Second for loop:
        >>> # Here the ``mask_temporary`` will be moved to the right for each itteration.
        >>> # The ``+`` opertaor will be used (1+1=1, 1+0=1 , 0+0=0).
        >>> mask_result     [0,0,0,0,0,0,0,0,0,0,0,0]       # mask_result starts with 0
        >>>                 - - - - - - - - - - - -
        >>> mask_temporary  [0,0,1,0,0,0,0,1,1,0]
        >>>         ""        [0,0,1,0,0,0,0,1,1,0]
        >>>         ""          [0,0,1,0,0,0,0,1,1,0]
        >>>                   - - - - - - - - - - - -
        >>> mask_result =   [0,0,1,1,1,0,0,1,1,1,1,0]
        >>> da_mask =       [1,0,1,1,1,0,0,1,1,1,1,0]

    Examples
    --------
        >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
        >>> consecutive_events(da_mask, min_duration = 3)
    """
    # make sure that the mask is boolean
    if (
        np.issubdtype(da_mask.dtype, float)
        or np.issubdtype(da_mask.dtype, int)
        or np.issubdtype(da_mask.dtype, bool)
    ):
        da_mask = da_mask.astype(bool)
    else:
        raise ValueError(f"da_mask must be boolean but is type: {da_mask.dtype}")

    if da_mask[axis].ndim != 1:
        raise ValueError(f"axis must be one dimensional but is {da_mask[axis].ndim}")

    # try:
    #     da_mask = da_mask.astype(bool)
    # except Exception as e:
    #     raise ValueError(f"da_mask must be boolean. Error: {e}")
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
    This function calculates the mask of consecutive events with a duration of at least
    ´´min_duration´´(default 1) It runs full on numpy.

    Note
    ----
    The provided array will be converted into a boolean array.
    This means that all values that are not 0 will be converted to True.
    Except for np.nan values or any other value that is not convertable to bool.

    Parameters
    ----------
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

    Notes
    --------------------------
    (1 - True, 0 - False)

    Example setup
        >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
        >>> consecutive_events(da_mask, min_duration = 3)

    First for loop: mask_temporary
        >>> # It will create the following temporary mask_index (start index always moved to the right.)
        >>> [1,0,1,1,1,0,0,1,1,1]    # [1,0,1,1,1,0,0,1,1,1]
        >>> [0,1,1,1,0,0,1,1,1,1]    #   [0,1,1,1,0,0,1,1,1,1]
        >>> [1,1,1,0,0,1,1,1,1,0]    #     [1,1,1,0,0,1,1,1,1,0]
        >>>  - - - - - - - - - -     # with & operator the following will be done:
        >>> [0,0,1,0,0,0,0,1,1,0]    # <- mask_temporary

    Second for loop:
        >>> # Here the ``mask_temporary`` will be moved to the right for each itteration.
        >>> # The ``+`` opertaor will be used (1+1=1, 1+0=1 , 0+0=0).
        >>> mask_result     [0,0,0,0,0,0,0,0,0,0,0,0]       # mask_result starts with 0
        >>>                 - - - - - - - - - - - -
        >>> mask_temporary  [0,0,1,0,0,0,0,1,1,0]
        >>>         ""        [0,0,1,0,0,0,0,1,1,0]
        >>>         ""          [0,0,1,0,0,0,0,1,1,0]
        >>>                   - - - - - - - - - - - -
        >>> mask_result =   [0,0,1,1,1,0,0,1,1,1,1,0]
        >>> da_mask =       [1,0,1,1,1,0,0,1,1,1,1,0]

    Examples
    --------
        >>> da_mask = [1,0,1,1,1,0,0,1,1,1,1,0]
        >>> consecutive_events(da_mask, min_duration = 3)
    """

    # make sure that the mask is boolean
    # try:
    if (
        np.issubdtype(mask.dtype, float)
        or np.issubdtype(mask.dtype, int)
        or np.issubdtype(mask.dtype, bool)
    ):
        mask = mask.astype(bool)
    else:
        raise ValueError(f"da_mask must be boolean but is type: {mask.dtype}")

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
    Select an individual cloud from the individual cloud dataset based on its
    ``cloud_id`` The data for the individual cloud is extracted from the clouds dataset
    ``ds_clouds``.

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


def __select_allowed_dropsonde_by_time__(
    ds_distance: xr.Dataset,
    cloud_time: np.datetime64,
    max_temporal_distance: np.timedelta64,
    max_spatial_distance: float,
    index_ds_clouds: str,
    index_ds_dropsonde: str,
    name_dt: str,
    name_dx: str,
) -> xr.DataArray:
    """
    Returns the allowed times of the dropsondes for a given cloud time. The allowed
    times are the times of the dropsondes that are within the maximum temporal and
    spatial distance of the cloud.

    Parameters
    ----------
    ds_distance : xr.Dataset
        Dataset containing distance data between clouds and dropsondes.
        It is a 2D look up table which constains the temporal and spatial distances between clouds and dropsondes.
        It choords are:
        - ``index_ds_clouds``: time of the clouds
        - ``index_ds_dropsonde``: time of the dropsondes
        Its data variables are:
        - ``name_dt``: (index_ds_clouds, index_ds_dropsonde)
            temporal distance between clouds and dropsondes
        - ``name_dx``: (index_ds_clouds, index_ds_dropsonde)
            spatial distance between clouds and dropsondes
    cloud_time : np.datetime64
        Time of the cloud.
    max_temporal_distance : np.timedelta64
        Maximum allowed temporal distance between a clouds and a dropsonde for them to be considered a match.
    max_spatial_distance : float
        Units are in kilometers.
        Maximum allowed spatial distance (in the same units as ds_distance) between a cloud and a dropsonde for them to be considered a match.
    index_ds_clouds : str
        Index of the clouds dataset.
    index_ds_dropsonde : str
        Index of the dropsonde dataset.
    name_dt : str
        Name of the temporal distance data variable in the distance dataset ``ds_distance``.
    name_dx : str
        Name of the spatial distance data variable in the distance dataset ``ds_distance``.

    Returns
    -------
    xr.DataArray
        Allowed times of the dropsondes for a given cloud time.
        The allowed times are the times of the dropsondes that are within the
        maximum temporal and spatial distance of the cloud.

    Examples
    --------
    """

    distance_selection = ds_distance.sel(
        {
            index_ds_clouds: cloud_time,
        }
    )
    return distance_selection[index_ds_dropsonde].where(
        # temporal distance
        (np.abs(distance_selection[name_dt].data) <= max_temporal_distance)
        # spatial distance
        & (distance_selection[name_dx] <= max_spatial_distance),
        drop=True,
    )


def match_clouds_and_dropsondes(
    ds_clouds: xr.Dataset,
    ds_sonde: xr.Dataset,
    ds_distance: xr.Dataset,
    max_temporal_distance: np.timedelta64 = np.timedelta64(1, "h"),
    max_spatial_distance: float = 100,
    dim_in_clouds: str = "time",
    dim_in_dropsondes: str = "time",
    index_ds_clouds: str = "time_identified_clouds",
    index_ds_dropsonde: str = "time_drop_sondes",
    name_dt: str = "temporal_distance",
    name_dx: str = "spatial_distance",
    dask_compute: bool = True,
):
    """
    Returns the Subdataset of dropsondes based on their spatial and temporal distances
    to the individual cloud. The data for the individual clouds can be extracted from
    the individual clouds dataset. The dropsonde dataset is the Level 3 dropsonde
    dataset.

    Note
    ----
        - This function uses a simple for loop to loop over the clouds in the clouds dataset.
        - Therefore it is not very efficient for many clouds.
        - If the values of ``dim_in_clouds`` from the clouds dataset are not ALL in the ``index_ds_clouds`` from the distance dataset.
        - If the values of ``dim_in_clouds`` from the clouds dataset and ``index_ds_clouds`` from the distance dataset are not equal.

    Parameters
    ----------
    ds_clouds : xr.Dataset
        Dataset containing data for multiple clouds from the individual clouds dataset.
        It is indexed by ``dim_in_clouds``. Default is 'time'.
    ds_sonde : xr.Dataset
        Dataset containing dropsonde data.
        It is indexed by ``dim_in_dropsondes``. Default is 'time'.
    ds_distance : xr.Dataset
        Dataset containing distance data between clouds and dropsondes.
        It is a 2D look up table which constains the temporal and spatial distances between clouds and dropsondes.
        It choords are: ``index_ds_clouds``: time of the clouds, ``index_ds_dropsonde``: time of the dropsondes
        Its data variables are: ``name_dt``: (index_ds_clouds, index_ds_dropsonde) temporal distance between clouds and dropsondes ``name_dx``: (index_ds_clouds, index_ds_dropsonde) spatial distance between clouds and dropsondes
    max_temporal_distance : np.timedelta64, optional
        Maximum allowed temporal distance between a clouds and a dropsonde for them to be considered a match.
        Default is 1 hour.
    max_spatial_distance : float, optional
        Units are in kilometers.
        Maximum allowed spatial distance (in the same units as ds_distance) between a cloud and a dropsonde for them to be considered a match.
        Default is 100 (km).
    index_ds_clouds : str, optional
        Index of the clouds dataset. Default is 'time'.
    index_ds_dropsonde : str, optional
        Index of the dropsonde dataset. Default is 'time_drop_sondes'.
    name_dt : str, optional
        Name of the temporal distance data variable in the distance dataset ``ds_distance``.
        Default is 'temporal_distance'.
    name_dx : str, optional
        Name of the spatial distance data variable in the distance dataset ``ds_distance``.
        Default is 'spatial_distance'.
    dask_compute : bool, optional
        If True, the output dataset is computed. Default is True.
        Dask will not be used if compute is False and then be lazy.

    Returns
    -------
    xr.Dataset
        A subset of the dropsonde dataset that matches with the clouds dataset based on the specified spatial and temporal distances.
        The dataset contains the same variables as the input dropsonde dataset.

    Raises
    ------
    ValueError
        If the values of ``dim_in_clouds`` from the clouds dataset are not ALL in the ``index_ds_clouds`` from the distance dataset.
        If the values of ``dim_in_clouds`` from the clouds dataset and ``index_ds_clouds`` from the distance dataset are not equal.

    Examples
    --------
    Example visualisation

    >>> # Example setup
    >>> # For a max dt = 5 hours
    >>> # For a max dh = 3 km
    >>> # The datasets below can be summarized visually as follows:

    >>> # 0    5    10   15   20  |1    6    11   # Time in hours
    >>> # ---S--M--E--------------|--S--M--E----  # Cloud start, middle, end time
    >>> # D----D----D----D----D---|D----D----D--  # Dropsonde
    >>> # 1----1----3----3----5---|3----1----8--  # Distance dropsonde to clouds in km (same for both clouds)

    >>> # 0    5    10   15   20  |1    6    11   # Time in hours
    >>> # F----T----T----F----F---|T----T----F--  # Dropsondes close to cloud T for true F for false
    >>> # F----T----T----F----F---|F----F----F--  # Dropsondes close to cloud 0
    >>> # F----T----T----F----F---|T----T----F--  # Dropsondes close to cloud 1

    Example datasets

    >>> ds_clouds = xr.Dataset(
    ...     {
    ...         "cloud_id": (("time",), [0, 1]),
    ...         "start": (("time",), pd.date_range("2020-01-01 3:00", periods=2, freq="D")),
    ...         "end": (("time",), pd.date_range("2020-01-01 9:00", periods=2, freq="D")),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01 6:00", periods=2, freq="D"),
    ...     },

    >>> ds_sonde = xr.Dataset(
    ...     {
    ...         "temp": (("time",), np.arange(8)),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01 0:00", periods=8, freq="5H"),
    ...     },
    ... )

    >>> ds_distance = xr.Dataset(
    ...     {
    ...         "spatial_distance": (
    ...             ("time_identified_clouds", "time_drop_sondes"),
    ...             np.array([
    ...                 [1, 1, 3, 3, 5, 3, 1, 8],
    ...                 [1, 1, 3, 3, 5, 3, 1, 8],
    ...             ], dtype="int"),
    ...         ),
    ...         "temporal_distance": (
    ...             ("time_identified_clouds", "time_drop_sondes"),
    ...             np.array([
    ...                     [  6,   1,  -4,  -9, -14, -19, -24, -29],
    ...                     [ 30,  25,  20,  15,  10,   5,   0, -5]
    ...                 ], dtype="timedelta64[h]"),
    ...         ),
    ...     },
    ...     coords={
    ...         "time_drop_sondes": ds_sonde.time.data,
    ...         "time_identified_clouds": ds_clouds.time.data,
    ...     },
    ... )

    >>> result = match_clouds_and_dropsondes(
    ...     ds_clouds = ds_clouds,
    ...     ds_sonde = ds_sonde,
    ...     ds_distance = ds_distance,
    ...     dim_in_dropsondes = "time",
    ...     index_ds_dropsonde = "time_drop_sondes",
    ...     index_ds_clouds = "time_identified_clouds",
    ...     max_temporal_distance = np.timedelta64(5, "h"),
    ...     max_spatial_distance = 3,
    ... )
    >>> print(result)
    <xarray.Dataset>
    Dimensions:  (time: 4)
    Coordinates:
    * time     (time) datetime64[ns] 2020-01-01T05:00:00 ... 2020-01-02T06:00:00
    Data variables:
        temp     (time) int64 1 2 5 6
    """

    # make sure that all values of the clouds dataset are in the distance dataset
    if False == np.all(
        np.isin(
            ds_clouds[dim_in_clouds].data,
            ds_distance[index_ds_clouds].data,
        )
    ):
        assert_message = "All 'dim_in_clouds' values from the Clouds dataset must be in the 'index_ds_clouds' from the distance dataset!"
        raise ValueError(assert_message)
    # make sure that all values of the dropsonde dataset are in the distance dataset
    try:
        np.testing.assert_equal(ds_distance[index_ds_dropsonde].data, ds_sonde[dim_in_dropsondes].data)
    except AssertionError as e:
        assert_message = "The values of 'dim_in_dropsondes' from dropsonde dataset and 'index_ds_dropsonde' from distance dataset must be equal!"
        raise ValueError(assert_message)

    # create a list of time arrays for each cloud
    time_list = []

    # If the clouds dataset has multiple clouds, we need to loop over them
    # this also works for one cloud, as long as the dim_in_clouds is not dimension less (ndim = 0)
    if ds_clouds[dim_in_clouds].ndim != 0:
        for cloud_time in ds_clouds[dim_in_clouds].data:
            allowed_dropsonde_times = __select_allowed_dropsonde_by_time__(
                ds_distance=ds_distance,
                cloud_time=cloud_time,
                max_temporal_distance=max_temporal_distance,
                max_spatial_distance=max_spatial_distance,
                index_ds_clouds=index_ds_clouds,
                index_ds_dropsonde=index_ds_dropsonde,
                name_dt=name_dt,
                name_dx=name_dx,
            )
            time_list.append(allowed_dropsonde_times)
    else:
        # There is an issue when selecting a single cloud based on its time.
        # This leads to the fact that the time dimension is kinda dropped.
        # Thus if ndim is 0, we need to hand the __select_allowed_dropsonde_times__ function a single time value.
        allowed_dropsonde_times = __select_allowed_dropsonde_by_time__(
            ds_distance=ds_distance,
            cloud_time=ds_clouds[dim_in_clouds].data,
            max_temporal_distance=max_temporal_distance,
            max_spatial_distance=max_spatial_distance,
            index_ds_clouds=index_ds_clouds,
            index_ds_dropsonde=index_ds_dropsonde,
            name_dt=name_dt,
            name_dx=name_dx,
        )
        time_list.append(allowed_dropsonde_times)

    # concatenate the list of time arrays and sort them
    time_array = xr.concat(time_list, dim=index_ds_dropsonde)
    # make sure to only use unique times
    time_array = np.unique(time_array)
    time_array = np.sort(time_array)

    if dask_compute is True:
        return ds_sonde.sel({dim_in_dropsondes: time_array}, drop=True).compute()
    else:
        return ds_sonde.sel({dim_in_dropsondes: time_array}, drop=True)


def match_clouds_and_cloudcomposite(
    ds_clouds: xr.Dataset,
    ds_cloudcomposite: xr.Dataset,
    dim: str = "time",
    var_name_start: str = "start",
    var_name_end: str = "end",
    dask_compute: bool = True,
):
    """
    Returns the subdataset of the cloud composite dataset which is part of multiple
    individual cloud. The selection is performed purely based on the start and end time
    of the individual cloud.

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

    Examples
    --------
    >>> ds_cloudcomposite = xr.Dataset(
    ...     {
    ...         "cloud_composite": (("time", "latitude", "longitude"), np.random.rand(5, 5, 5)),
    ...     },
    ...     coords={
    ...         "time": pd.date_range("2020-01-01", periods=5),
    ...         "latitude": np.arange(5),
    ...         "longitude": np.arange(5),
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
    Dimensions:          (latitude: 5, longitude: 5, time: 3)
    Coordinates:
        * time             (time) datetime64[ns] 2020-01-01 2020-01-02 2020-01-03
        * latitude              (latitude) int64 0 1 2 3 4
        * longitude              (longitude) int64 0 1 2 3 4
    Data variables:
        cloud_composite  (time, latitude, longitude) float64 0.548 0.592 0.046 0.607 ... 0.236 0.604 0.959 0.22
    """

    # create a list of time arrays for each cloud
    time_list = []
    if ds_clouds[dim].ndim != 0:
        for start_time, end_time in zip(
            ds_clouds[var_name_start],
            ds_clouds[var_name_end],
        ):
            time_list.append(ds_cloudcomposite.sel({dim: slice(start_time, end_time)})[dim])
    else:
        start_time = ds_clouds[var_name_start]
        end_time = ds_clouds[var_name_end]
        time_list.append(ds_cloudcomposite.sel({dim: slice(start_time, end_time)})[dim])
    # concatenate the list of time arrays and sort them
    time_array = xr.concat(time_list, dim=dim)
    time_array = time_array.sortby(dim)

    if dask_compute is True:
        return ds_cloudcomposite.sel({dim: time_array}).compute()
    else:
        return ds_cloudcomposite.sel({dim: time_array})
