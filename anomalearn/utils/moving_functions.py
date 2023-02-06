import numpy as np


def mov_avg(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.
        
    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_avg : ndarray of shape (n_samples, n_features)
        The moving average time series with same shape as `x`.
    """
    x = np.array(x, dtype=np.longdouble)
    if x.ndim == 1:
        x = x.reshape((-1, 1))
    
    left = window // 2
    right = left - (window % 2 == 0)
    if clip == "left":
        left, right = right, left
    
    avg_series = np.zeros_like(x)
    for i in range(x.shape[0]):
        start = i - left if i - left >= 0 else 0
        end = i + 1 + right
        avg_series[i, :] = np.mean(x[start:end, :], axis=0)
    
    return avg_series


def mov_std(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.
        
    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_std : ndarray of shape (n_samples, n_features)
        The moving standard deviation time series with same shape as `x`.
    """
    x = np.array(x, dtype=np.longdouble)
    if x.ndim == 1:
        x = x.reshape((-1, 1))

    left = window // 2
    right = left - (window % 2 == 0)
    if clip == "left":
        left, right = right, left
    
    std_series = np.zeros_like(x)
    for i in range(x.shape[0]):
        start = i - left if i - left >= 0 else 0
        end = i + 1 + right
        std_series[i, :] = np.std(x[start:end, :], axis=0)
    
    return std_series
