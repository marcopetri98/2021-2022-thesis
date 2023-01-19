from typing import Tuple

import numpy as np
from sklearn.utils import check_array

from .. import IShapeChanger, SavableModel


class SlidingWindowForecast(IShapeChanger, SavableModel):
    """Preprocessing object transforming the input using the sliding window technique.
    
    The object takes as input a time series, thus an array-like with two
    dimensions.
    
    Parameters
    ----------
    window : int
        It is the window to be used while doing the sliding window.
    
    stride : int, default=1
        It is the stride to be used while doing the sliding window.
        
    forecast : int, default=1
        It is the number of forecast to consider in building the sliding windows.
    """
    def __init__(self, window: int,
                 stride: int = 1,
                 forecast: int = 1):
        super().__init__()
        
        self.window = window
        self.stride = stride
        self.forecast = forecast
        
    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        y
            Ignored.
            
        Returns
        -------
        x_new : array-like
            The sliding windows of the time series with given window and stride.
            
        y_new : array-like
            The windows containing the next `forecast` points after `x_new`.
        """
        check_array(x)
        x = np.array(x)
    
        window_data = None
        window_target = None
        for i in range(0, x.shape[0] - self.window - self.forecast + 1, self.stride):
            new_data = x[i:i + self.window].reshape((1, self.window, x.shape[1]))
            new_target = x[i + self.window:i + self.window + self.forecast].reshape((1, self.forecast, x.shape[1]))
            
            if window_data is None:
                window_data = new_data.copy()
                window_target = new_target.copy()
            else:
                window_data = np.concatenate((window_data, new_data))
                window_target = np.concatenate((window_target, new_target))
                
        return window_data, window_target
