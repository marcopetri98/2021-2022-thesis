from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_time_series_decomposition(original,
                                   seasonal,
                                   trend,
                                   residual,
                                   fig_size: Tuple = (16, 16)) -> None:
    """Plots in a single figure the original, seasonal, trend and residual.
    
    Parameters
    ----------
    original : array-like
        The original time series.
    
    seasonal : array-like
        The seasonal component of the time series.
    
    trend : array-like
        The trend component of the time series.
    
    residual : array-like
        The residual component of the time series.

    fig_size : Tuple
        The dimension of the figure.

    Returns
    -------
    None
    """
    original = np.array(original)
    seasonal = np.array(seasonal)
    trend = np.array(trend)
    residual = np.array(residual)
    
    x_values = np.arange(original.shape[0])
    
    fig, axs = plt.subplots(4, 1, figsize=fig_size)
    plt.title("Decomposition results")
    axs[0].set_title("Observed")
    axs[0].plot(x_values, original, linewidth=0.5)
    axs[1].set_title("Trend")
    axs[1].plot(x_values, trend, linewidth=0.5)
    axs[2].set_title("Seasonal")
    axs[2].plot(x_values, seasonal, linewidth=0.5)
    axs[3].set_title("Residual")
    axs[3].scatter(x_values, residual, linewidths=0.5)
    plt.show()
