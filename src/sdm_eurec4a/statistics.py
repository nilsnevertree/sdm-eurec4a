import warnings

from typing import Tuple, Union

import numpy as np
import xarray as xr

from scipy import fftpack


def RMSE(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the root mean squared error (RMSE) between two arrays.

    Parameters:
    -----------
    x : array-like
        The predicted values.
    y : array-like
        The true values.

    Returns:
    --------
    float
        The root mean squared error between `x` and `y`.

    Examples:
    ---------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> RMSE(x, y)
    2
    """
    return np.sqrt(np.mean((x - y) ** 2))


def xarray_RMSE(
    x: Union[xr.Dataset, xr.DataArray],
    y: Union[xr.Dataset, xr.DataArray],
    dim: str = "time",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Calculate the root mean squared error (RMSE) between two arrays.

    Note:
    - Nan values will be fully ignored by this function!

    Parameters:
    -----------
    x : xr.Dataset or xr.DataArray
        The predicted values.
    y : xr.Dataset or xr.DataArray
        The true values.
    dim : str
        Dimension for which the RMSE shall be computed.

    Returns:
    --------
    xr.Dataset or xr.DataArray
        The root mean squared error between `x` and `y`.

    Examples:
    ---------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 4, 6])
    >>> RMSE(x, y)
    2
    """
    return np.sqrt(((x - y) ** 2).mean(dim=dim))


def coverage(x: np.ndarray, P: np.ndarray, y: np.ndarray, stds: float = 0.64) -> np.ndarray:
    """
    Calculate the coverage of a prediction interval.

    Parameters:
    -----------
    x : array-like
        The predicted values.
    P : array-like
        The variance or uncertainty associated with the predicted values.
    y : array-like
        The true values.
    stds : float, optional
        The number of standard deviations to consider for the prediction interval.
        Default is 0.64, corresponding to approximately 50% coverage.

    Returns:
    --------
    ndarray
        Boolean array indicating whether the true values fall within the prediction interval.

    Examples:
    ---------
    >>> x = np.array([1, 2, 3])
    >>> P = np.array([1, 2, 3])
    >>> y = np.array([2, 5, 7])
    >>> coverage(x, P, y)
    array([ True, False, False])
    """
    return (y >= x - stds * np.sqrt(P)) & (y <= x + stds * np.sqrt(P))


def coverage_prob(x: np.ndarray, P: np.ndarray, y: np.ndarray, stds: float = 0.64) -> np.ndarray:
    """
    Calculate the coverage probability of a prediction interval.

    Parameters:
    -----------
    x : array-like
        The predicted values.
    P : array-like
        The variance or uncertainty associated with the predicted values.
    y : array-like
        The true values.
    stds : float, optional
        The number of standard deviations to consider for the prediction interval.
        Default is 0.64, corresponding to approximately 50% coverage.

    Returns:
    --------
    float
        The coverage probability of the prediction interval.

    Examples:
    ---------
    >>> x = np.array([1, 2, 3])
    >>> P = np.array([1, 2, 3])
    >>> y = np.array([2, 5, 7])
    >>> coverage_prob(x, P, y)
    0.3333333333333333
    """
    res = coverage(x=x, P=P, y=y, stds=stds)
    return np.sum(res) / np.size(res)


def xarray_coverage_prob(
    x: xr.DataArray,
    P: xr.DataArray,
    y: xr.DataArray,
    stds: float = 0.64,
    dim: str = "time",
) -> xr.DataArray:
    """
    Calculate the coverage probability of a prediction interval. Note that x
    and y should contain the same dimensions and should be of same shape.

    Parameters:
    -----------
    x : xr.DataArray
        The predicted values.
    P : xr.DataArray
        The variance or uncertainty associated with the predicted values.
    y : xr.DataArray
        The true values.
    stds : float, optional
        The number of standard deviations to consider for the prediction interval.
        Default is 0.64, corresponding to approximately 50% coverage.
    dim : str
        Dimension for which the coverage probability shall be computed.

    Returns:
    --------
    xr.DataArray
        The coverage probability of the prediction interval.

    Examples:
    ---------
    >>> x = np.array([1, 2, 3])
    >>> P = np.array([1, 2, 3])
    >>> y = np.array([2, 5, 7])
    >>> coverage_prob(x, P, y)
    0.3333333333333333
    """
    res = coverage(x=x, P=P, y=y, stds=stds)
    return res.sum(dim=dim) / np.size(res[dim])


def broadcast_along_axis_as(x: np.ndarray, y: np.ndarray, axis: int):
    """
    Broadcasts 1D array x to an array of same shape as y, containing the given
    axis. The length of x need to be the same as the length of y along the
    given axis. Note that this is a broadcast_to, so the return is a view on x.
    Based on the answer at https://stackoverflow.com/a/62655664/16372843.

    Parameters
    ----------
    x: array
        Array of dimension 1 which should be broadcasted for a specific axis.
    x: array
        Array of dimension 1 which should be broadcasted for a specific axis.
    axis: int
        Axis along which the arrays align.

    Returns
    -------
    array
        Array containing values along provided axis as x but with shape y.
        Note that this is a broadcast_to, so the return is a view on x.
    """
    # shape check
    if axis >= y.ndim:
        raise np.AxisError(axis, y.ndim)
    if x.ndim != 1:
        raise ValueError(f"ndim of 'x' : {x.ndim} must be 1")
    if x.size != y.shape[axis]:
        raise ValueError(
            f"Length of 'x' must be the same as y.shape along the axis. But found {x.size}, {y.shape[axis]}, axis= {axis}"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(y, y.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    res = np.broadcast_to(x, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    res = np.swapaxes(res, y.ndim - 1, axis)
    return res


def gaussian_kernel_1D(
    x: np.ndarray,
    center_idx: int,
    axis: int = 0,
    sigma: float = 100,
    same_output_shape: bool = False,
) -> np.ndarray:
    """
    Creates a Gaussian weights for a N dimensional array x centered at index
    center_idx along specified axis.

    Parameters
    ----------
    x: np.ndarray
        Array of dimension N with length l along provided axis.
    center_idx: int
        Index of the center of the gaussian kernel along provided axis.
    axis: int
        Axis along which the weights shall be computed.
    sigma: float
        Standard deviation as the sqrt(variance) of the gaussian distribution.
        Default to 10
    same_output_shape : bool
        Sets if the output array should be of shape.
        Output array is 1D array if False.
        Output array of same shape as x if True.
        Then the weights will be along the provided axis

    Returns
    -------
    np.ndarray
        Array containing the weights of the kernel.
        If output is 1-dimensional, along this axis.
        If output in N-dimensional, along provided axis.
        See also 'same_output_shape'.
    """

    index_array = np.arange(x.shape[axis])

    # apply the Gaussian kernel
    kernel = np.exp(-0.5 * ((index_array - center_idx) ** 2) / (sigma**2))
    kernel = kernel / np.sum(kernel)
    # normalize the weights
    if not same_output_shape:
        return kernel
    else:
        return broadcast_along_axis_as(kernel, x, axis=axis)


def __normalize_minmax__(
    self: Union[xr.DataArray, xr.Dataset], dim: Union[None, str] = None
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Normalize the array using min-max normalization.

    Returns:
        np.ndarray: Normalized array using min-max normalization.
    """
    if dim is None:
        return (self - self.min()) / (self.max() - self.min())
    else:
        return (self - self.min(dim=dim)) / (self.max(dim=dim) - self.min(dim=dim))


def __normalize_mean__(
    self: Union[xr.DataArray, xr.Dataset], ddof: int = 0, dim: Union[None, str] = None
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Normalize the array using mean normalization.

    Parameters:
        ddof (int, optional): Delta degrees of freedom. The divisor used in the calculation is N - ddof, where N represents the number of elements. Default is 0.

    Returns:
        np.ndarray: Normalized array using mean normalization.
    """
    if dim is None:
        return (self - self.mean()) / self.std(ddof=ddof)
    else:
        return (self - self.mean(dim=dim)) / self.std(ddof=ddof, dim=dim)


def __normalize_oneone__(
    self: Union[xr.DataArray, xr.Dataset], dim: Union[None, str] = None
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Normalize the array to the range [-1, 1].

    Returns:
        np.ndarray: Normalized array to the range [-1, 1].
    """
    return __normalize_minmax__(self=self, dim=dim) * 2 - 1


def normalize(
    x: Union[xr.Dataset, xr.DataArray, np.ndarray],
    method: str = "oneone",
    ddof: int = 0,
    dim: Union[None, str] = None,
) -> Union[xr.Dataset, xr.DataArray, np.ndarray]:
    """
    Normalize the input array using the specified method.

    Parameters:
        x (Union[xr.Dataset, xr.DataArray, np.ndarray]): The input array to be normalized.
        method (str, optional): The normalization method to use.
            - a) "MinMax" or "minmax" or "01" or "0-1": Min-max normalization. Values are scaled to the range [0, 1].
            - b) "Mean" or "mean" or "norm": Mean normalization. Values are centered around the mean and scaled by the standard deviation.
            - c) "OneOne" or "oneone" or "11" or "1-1": Scaling to the range [-1, 1] using min-max normalization.
            - Default of `method` is "oneone".
        ddof (int, optional): Delta degrees of freedom.
            - The divisor used in the calculation is N - ddof, where N represents the number of elements.
            - Default is 0.
            - Only used with b).

    Returns:
        np.ndarray: The normalized array.

    Raises:
        AssertionError: If an invalid normalization method is provided.

    Example:
    >>> ds = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), temperature_data)},
        coords={
            "time": pd.date_range("2022-01-01", periods=365),
            "latitude": [30, 40, 50],
            "longitude": [-120, -110, -100],
        },
    )
    >>> normalized_ds = normalize(ds, method="minmax")
    >>> print(f"Normalized dataset:\n{normalized_ds}")
    """
    if method in ["MinMax", "minmax", "01", "0-1"]:
        return __normalize_minmax__(x, dim=dim)
    elif method in ["Mean", "mean", "norm"]:
        return __normalize_mean__(x, ddof=ddof, dim=dim)
    elif method in ["OneOne", "oneone", "11", "1-1"]:
        return __normalize_oneone__(x, dim=dim)
    else:
        assert False, f"Invalid normalization method: {method}"


def autocorr(ds: xr.DataArray, lag: int = 0, dim: str = "time"):
    """
    Compute the lag-N autocorrelation using Pearson correlation coefficient.

    Parameters:
        ds (xr.DataArray): The object for which the autocorrelation shall be computed.
        lag (int, optional): Number of lags to apply before performing autocorrelation. Default is 0.
        dim (str, optional): Dimensino along which the autocorrelation shall be performed. Default is "time".

    Returns:
        float: The autocorrelation value.

    Example:
    >>> ds = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), temperature_data)},
        coords={
            "time": pd.date_range("2022-01-01", periods=365),
            "latitude": [30, 40, 50],
            "longitude": [-120, -110, -100],
        },
    )
    >>> # compute 30 day lagged auto-correlation
    >>> autocorr_value = autocorr(ds["temperature"], lag=30, dim="time")
    >>> print(f"Autocorrelation value: {autocorr_value}")
    """
    if isinstance(ds, xr.DataArray):
        return xr.corr(ds, ds.shift({f"{dim}": lag}))
    else:
        raise NotImplementedError(f"Not implemented for type: {type(ds)}.")


def crosscorr(ds1: xr.DataArray, ds2: xr.DataArray, lag: int = 0, dim: str = "time"):
    """
    Compute the lag-N cross-correlation using Pearson correlation coefficient of ds1 on ds2.
    ds2 will be shihfted by ``lag`` timesteps.

    Parameters:
        ds1 (xr.DataArray): First array for the cross-correlation.
        ds2 (xr.DataArray): Second array for the cross-correlation. This array will be shifted.
        lag (int, optional): Number of lags to apply before performing autocorrelation. Default is 0.
        dim (str, optional): Dimensino along which the autocorrelation shall be performed. Default is "time".

    Returns:
        xr.DataArray: Containing the result of the cross-correlation.

    Example:
    >>> ds1 = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), temperature_data)},
        coords={
            "time": pd.date_range("2022-01-01", periods=365),
            "latitude": [30, 40, 50],
            "longitude": [-120, -110, -100],
        },
    )
    >>> ds2 = xr.Dataset(
        {"precipitation": (("time", "latitude", "longitude"), precipitation_data)},
        coords={
            "time": pd.date_range("2022-01-01", periods=365),
            "latitude": [30, 40, 50],
            "longitude": [-120, -110, -100],
        },
    )
    >>> # compute 30 day lagged cross correlation
    >>> crosscorr_value = crosscorr(
            ds1 = ds1["temperature"],
            ds2 = ds2["precipitation"],
            lag=30,
            dim="time",
            )
    >>> print(f"Cross-correlation value: {crosscorr_value}")
    """
    if isinstance(ds1, xr.DataArray) and isinstance(ds2, xr.DataArray):
        return xr.corr(ds1, ds2.shift({f"{dim}": lag}), dim=dim)
    else:
        raise NotImplementedError(f"Not implemented for type: {type(ds1)} and {type(ds2)}.")
