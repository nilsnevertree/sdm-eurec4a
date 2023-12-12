import warnings

from typing import Tuple, Union

import numpy as np
import xarray as xr

from scipy import fftpack


def my_mean(x: np.ndarray, axis: None = None, **kwargs) -> np.ndarray:
    """
    Calculate the mean of a numpy array using np.mean This functions only
    purpose is to check the CI workflows of the repository.

    Parameters
    ----------
    x: array
        Array for which the mean shall be calculated.

    Returns
    -------
    array
        Mean of input array x.
        ndim=1
    """

    return np.mean(x, axis=axis, **kwargs)


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


def gaussian_weights_2D(x: np.ndarray, y: np.ndarray, axis: int = 0, alpha: float = 0.2) -> np.ndarray:
    """
    Creates a Gaussian weights for a 2D-array x centered at positions given by
    y. The weights will be computed along the specified axis.

    Parameters
    ----------
    x: array
        Array of dimension (n, m).
        x.ndim should be 2.
    y: array
        Array of dimension (n, ).
        y.ndim should be 2 or 1.
    alpha: float
        Alpha of the gaussian distribution.
        Default to 0.2

    Returns
    -------
    array
        gaussian weights
        dimension as x
    """
    raise NotImplementedError("Function not fully implemented")
    try:
        assert np.array_equal(np.ndim(x), np.ndim(y))
    except Exception:
        if axis == 0:
            y = y[:, np.newaxis]
        elif axis == 1:
            y = y[np.newaxis, :]
    # apply the Gaussian kernel
    weights = np.exp((-((x - y) ** 2)) / (2 * alpha**2))

    # normalize the weights
    weights = weights / np.sum(weights)

    # return the weights
    return weights


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

    The equation used is:
    :math: `e^{-{\frac {1}{2}}\left({\frac {x-\center_idx}{\sigma }}\right)^{2}}}`

    Parameters
    ----------
    x: array
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
        - Output array is 1D array if False.
        - Output array of same shape as x if True.
          Then the weights will be along the provided axis

    Returns
    -------
    array
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


def ordered_like(a: list, b: list) -> list:
    """
    Sorts the elements in list 'a' based on their presence in list 'b' while
    preserving the order.

    Args:
        a (list): The list of elements to be sorted.
        b (list): The reference list used for sorting 'a'.

    Returns:
        list: A new list containing the elements from 'a' sorted based on their presence in 'b'.

    Example:
        a = ["x2", "x3"]
        b = ["x2", "z1", "x3", "z2"]
        result = ordered_like(a, b)
        print(result)
        # Output: ['x2', 'x3', 'z1', 'z2']
    """
    return sorted(a, key=lambda x: (x not in b, b.index(x) if x in b else False))


def assert_ordered_subset(a: list, b: list) -> list:
    """
    Asserts that list 'a' is a subset of list 'b' and that the elements in 'a'
    are ordered like in 'b'.

    Args:
        a (list): The list to be checked as a subset.
        b (list): The reference list used for the assertion.

    Raises:
        AssertionError: If 'a' is not a subset of 'b' or if the elements in 'a' are not ordered like in 'b'.

    Example:
        a = ["x2", "x3"]
        b = ["x2", "x1", "x3", "x4"]
        assert_ordered_subset(a, b)
        # No exception raised

        a = ["x2", "x3"]
        b = ["x3", "x1", "x2", "x4"]
        assert_ordered_subset(a, b)
        # AssertionError: a is not ordered like of b
    """
    assert set(a).issubset(b), "a is not a subset of b"
    assert a == [x for x in b if x in a], "a is not ordered like of b"


def kalman_single_forecast(
    S: np.ndarray, C: np.ndarray, M: np.ndarray, Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the forecast of a system using the Kalman equation.

    Parameters:
        S (np.ndarray): State matrix of shape [T x N].
        C (np.ndarray): Covariance matrix of shape [T x N x N].
        M (np.ndarray): Model matrix of shape [N x N] (2D case) or [T x N x N] (3D case).
        Q (np.ndarray): Model uncertainty matrix of shape [N x N] (2D case) or [T x N x N] (3D case).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the state forecast and covariance forecast.
            - state_forecast (np.ndarray): State forecast of shape [T x N].
            - covariance_forecast (np.ndarray): Covariance forecast of shape [T x N x N].

    Notes:
    - The function calculates the forecast of a system using the Kalman equations.
    - It can handle 2D and 3D cases of the model matrix M and model uncertainty matrix Q.
    - The state is updated using the Kalman equations: x(t+dt) = M x x(t), C(t+dt) = M x C(t) x M.T + Q.
    - The function supports different shapes of input matrices based on the dimensionality of M and Q.
    - For the 2D case, the input matrices have the following shapes:
        - S: [T x N]       (State Matrix)
        - C: [T x N x N]   (Covariance Matrix)
        - M: [N x N]       (Model matrix)
        - Q: [N x N]       (Model uncertainty matrix)
        - Update of S and Q:
            - S^u_{ij} = M_{jk} S_{ik}
            - C^u_{ijk} = M_{jk} C_{ijk} M_{kj} + Q_{jk}
    - For the 3D case, the input matrices have the following shapes:
        - S: [T x N]       (State Matrix)
        - C: [T x N x N]   (Covariance Matrix)
        - M: [T x N x N]   (Model matrix)
        - Q: [T x N x N]   (Model uncertainty matrix)
            - S^u_{ij} = M_{ijk} S_{ik}
            - C^u_{ijk} = M_{ijk} C_{ijk} M_{ikj} + Q_{ijk}
    - The state and covariance are updated based on the provided model matrices and uncertainty matrices.

    Raises:
        NotImplementedError: If the provided dimensionality of M and Q is not supported.
    """
    if M.ndim == Q.ndim == 2:
        state_forecast = np.einsum("jk,ik->ij", M, S)
        covariance_forecast = np.einsum("jk,ijk,kj->ijk", M, C, M) + Q
    elif M.ndim == Q.ndim == 3:
        state_forecast = np.einsum("ijk,ik->ij", M, S)
        covariance_forecast = np.einsum("ijk,ijk,ikj->ijk", M, C, M) + Q
    else:
        raise NotImplementedError(
            "For the provided case ndim of M : {np.ndim(M)}, Q : {np.ndim(Q)}, no Implementation is yet done."
        )
    return state_forecast, covariance_forecast


from typing import Union

import numpy as np
import xarray as xr


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


def compute_fft_spectrum(
    time: np.ndarray, signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    BETTER USE signal.welch( ''' from scipy import signal import
    matplotlib.pyplot as plt.

    fig, ax = plt.subplots(1,1)
    f, Pxx_den = signal.welch(x = oscillatory_data.DOT.values, fs=365.25/dt, nperseg=len(oscillatory_data.DOT.values))
    ax.loglog(f, Pxx_den, label = "DOT")

    dt = 30                     # days
    fs = 365.25/dt              # 1/years
    welch_window_width = 300    # years
    f, Pxx_den = signal.welch(
        x = oscillatory_data.DOT.values,
        fs=fs,
        window="hann",
        nperseg = int(welch_window_width*fs)
    )

    ax.loglog(f, Pxx_den, label = "DOT welch")
    '''

    Compute the Fast Fourier Transform (FFT) spectrum of the given signal.

    Parameters:
        time (array-like): The array representing the time domain values.
        signal (array-like): The array representing the signal values corresponding to `time`.

    Returns:
        tuple: A tuple containing the following FFT spectrum components:
            - frequencies (ndarray): The frequencies corresponding to the FFT spectrum.
            - spectrum (complex ndarray): The FFT spectrum of the input signal.
            - amplitude (ndarray): The absolute values of the FFT spectrum components (normalized).
            - min_frequency (float): The minimum frequency represented by the FFT spectrum.
            - max_frequency (float): The maximum frequency represented by the FFT spectrum.

    Examples:
        # Example 1: Generate and plot the FFT spectrum of a simple sine wave
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> time = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 20 * time)
        >>> frequencies, spectrum, amplitude, min_freq, max_freq = compute_fft_spectrum(time, signal)
        >>> plt.plot(frequencies, amplitude)
        >>> plt.xlabel('Frequency')
        >>> plt.ylabel('Amplitude')
        >>> plt.show()

        # Example 2: Analyze the spectrum of a sampled signal
        >>> sampling_rate = 1000  # Sampling rate
        >>> time = np.arange(0, 1, 1 / sampling_rate)
        >>> signal = np.sin(2 * np.pi * 50 * time) + np.sin(2 * np.pi * 120 * time)
        >>> frequencies, spectrum, amplitude, min_freq, max_freq = compute_fft_spectrum(time, signal)
        >>> plt.plot(frequencies, amplitude)
        >>> plt.xlabel('Frequency')
        >>> plt.ylabel('Amplitude')
        >>> plt.show()
    """
    message = " from scipy import signal\n#Use welch with a specified window width\nf, Pxx_den = signal.welch(x = data, fs=12, nperseg=window_length)"
    warnings.warn(message, DeprecationWarning)
    delta_t = time[1] - time[0]
    signal = signal[~np.isnan(signal)]

    num_samples = len(signal)
    time_values = np.arange(0, num_samples, delta_t)

    spectrum = fftpack.fft(signal)
    frequencies = np.linspace(0.0, 1.0 / (2.0 * delta_t), num_samples // 2)
    amplitude = 2.0 / num_samples * np.abs(spectrum[: num_samples // 2])
    min_frequency = 1 / num_samples
    max_frequency = 1 / (2 * delta_t)

    return frequencies, spectrum, amplitude, min_frequency, max_frequency
