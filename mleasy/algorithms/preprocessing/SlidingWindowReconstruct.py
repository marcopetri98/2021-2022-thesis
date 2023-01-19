from typing import Tuple

import numpy as np
from sklearn.utils import check_array

from .. import SavableModel, IShapeChanger


class SlidingWindowReconstruct(IShapeChanger, SavableModel):
    """Preprocessing object transforming the input using the sliding window technique.
    
    The object takes as input a time series, thus an array-like with two
    dimensions.
    
    Parameters
    ----------
    window : int
        It is the window to be used while doing the sliding window.
    
    stride : int, default=1
        It is the stride to be used while doing the sliding window.
    """
    def __init__(self, window: int,
                 stride: int = 1):
        super().__init__()
        
        self.window = window
        self.stride = stride
        
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
            Its shape is (n, window, features).
            
        y_new : array-like
            An array identical to `x_new`.
        """
        check_array(x)
        x = np.array(x)
    
        window_data = None
        for i in range(0, x.shape[0] - self.window + 1, self.stride):
            new_data = x[i:i + self.window].reshape((1, self.window, x.shape[1]))
            
            if window_data is None:
                window_data = new_data.copy()
            else:
                window_data = np.concatenate((window_data, new_data))
                
        return window_data, window_data.copy()
