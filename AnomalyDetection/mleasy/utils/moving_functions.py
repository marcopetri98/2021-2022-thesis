import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def mov_avg(x, window):
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like
        The original time series.

    window : int
        The window dimension.

    Returns
    -------
    mov_avg : ndarray
        The moving average time series.
    """
    x = np.array(x)
    is_multivariate = x.ndim != 1 and x.shape[1] > 1
    
    if is_multivariate:
        mov_avg_series = np.zeros((x.shape[0] - window + 1, x.shape[1]))
        for channel in mov_avg_series.shape[1]:
            mov_avg_series[:, channel] = np.convolve(x, [1 / window] * window, "valid")
    else:
        x = x.flatten()
        mov_avg_series = np.convolve(x, [1 / window] * window, "valid")
    
    return mov_avg_series


def mov_std(x, window):
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like
        The original time series.

    window : int
        The window dimension.

    Returns
    -------
    mov_std : ndarray
        The moving standard deviation time series.
    """
    x = np.array(x)
    is_multivariate = x.ndim != 1 and x.shape[1] > 1
    
    if is_multivariate:
        mov_std_series = np.zeros((x.shape[0] - window + 1, x.shape[1]))
        for channel in mov_std_series.shape[1]:
            sliding_windows = sliding_window_view(x, window)
            mov_std_series[:, channel] = np.array(list(map(lambda w: np.std(w), sliding_windows)))
    else:
        x = x.flatten()
        sliding_windows = sliding_window_view(x, window)
        mov_std_series = np.array(list(map(lambda w: np.std(w), sliding_windows)))
    
    return mov_std_series
