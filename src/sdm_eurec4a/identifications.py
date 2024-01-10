# %%

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def consecutive_events_xr(
    da_mask: xr.DataArray,
    min_duration: int = 1,
    axis: str = "time",
) -> xr.DataArray:
    """
    This function calculates the mask of consecutive events with a duration
    of at least ´´min_duration´´(default 1) It runs full on xarray and
    numpy.

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
    This function calculates the mask of consecutive events with a duration
    of at least ´´min_duration´´(default 1) It runs full on numpy.

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


# test_consecutive_events_np()

# example_mask  = np.array([
#                         [1,0,1,1,1,0,0,1,1,1,1,0],
#                         [1,1,1,0,0,0,0,1,1,1,1,0],
#                         [0,0,0,1,1,0,0,0,0,1,1,1],
#                         ], dtype=bool)

# plt.imshow(example_mask, alpha=0.5, cmap='Greys')
# plt.imshow(
#        consecutive_events_np(example_mask, min_duration = 3, axis = 1),
#         alpha=0.5, cmap='Reds',
# )
# %% DataArray Example
# example_mask = xr.DataArray([
#                         [1,0,1,1,1,0,0,1,1,1,1,0],
#                         [1,1,1,0,0,0,0,1,1,1,1,0],
#                         [0,0,0,1,1,0,0,0,0,1,1,1],
#                         ],
# )

# np.random.seed(0)
# example_mask = xr.DataArray(
#        np.random.choice([0, 1], size=(100,100,100), p=[1./3, 2./3])
# )

# result = consecutive_events_xr(example_mask, axis = 'dim_1', min_duration = 3)

# plt.imshow(example_mask, alpha=0.5, cmap='Greys')
# plt.imshow(result, alpha=0.5, cmap='Reds',)
# %%
