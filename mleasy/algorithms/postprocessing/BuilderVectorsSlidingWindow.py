from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .. import IShapeChanger, SavableModel, ICopyable
from ..preprocessing import SlidingWindowForecast, SlidingWindowReconstruct


class BuilderVectorsSlidingWindow(ICopyable, IShapeChanger, SavableModel):
    """Compute the vectors from the output of a model working on sliding windows.
    
    There is one vector for each timestamp of the original time series. The
    vectors are built as `v = [v11, v12, ..., v1l, v21, ..., v2l, ..., vd1, ...,
    vdl]`, where `vij` the prediction of feature `i` at time `t-j`. For the
    target vectors, for each `j != k` we have `vij` = `vik`. For reconstruction
    approaches `vij` is the reconstructed feature `i` by the window starting at
    time `t - j`. Also for reconstruction, the target vectors are such that for
    each `j != k` we have `vij` = `vik`.
    
    These two vectors can be used to compute an error vector and further compute
    a score for each timestamp as done in Malhotra et al.
    (https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf).
    
    Parameters
    ----------
    sliding_window : SlidingWindowForecast
        It is the sliding window that encoded the time series before giving it
        to the model.
        
    Attributes
    ----------
    _sliding_window : SlidingWindowForecast
        It is the reference to the sliding window object that has been passed at
        creation. It is not a copy of the object passed since the access to its
        properties are needed process data.
    """
    def __init__(self, sliding_window: SlidingWindowForecast | SlidingWindowReconstruct):
        super().__init__()
        
        self._sliding_window = sliding_window

    @property
    def sliding_window(self):
        return self._sliding_window

    def __repr__(self):
        return f"ErrorVectorsSlidingWindow(sliding_window={repr(self._sliding_window)})"
    
    def __str__(self):
        return f"Error vectors computation from {str(self._sliding_window)}"
        
    def __eq__(self, other):
        if not isinstance(other, BuilderVectorsSlidingWindow):
            return False
        
        return self._sliding_window is other._sliding_window

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> BuilderVectorsSlidingWindow:
        """Copies the object.
        
        Returns
        -------
        new_obj : BuilderVectorsSlidingWindow
            A new object identical to this.
        """
        new = BuilderVectorsSlidingWindow(sliding_window=self._sliding_window)
        return new
        
    def save(self, path: str,
             *args,
             **kwargs) -> Any:
        self._sliding_window.save(path)
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> Any:
        self._sliding_window = SlidingWindowForecast(1).load(path)
        return self
    
    def shape_change(self, x,
                     y=None,
                     *args,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Builds the vectors from model's predictions.
        
        Parameters
        ----------
        x : array-like of shape (n_windows, window/forecast, n_features)
            These are the values predicted by the model. Their appropriate name
            should be `y_hat`, but for API consistency `x` is left.
        
        y : array-like of shape (n_windows, window/forecast, n_features)
            The targets for the model outputs.

        Returns
        -------
        x_new : array-like
            The vectors with the predictions for each timestamp. It has two
            dimensions and the first is the number of points seen while building
            the sliding windows. The second dimension is the product between
            the features predicted/reconstructed and the window/forecasting.
            
        y_new : array-like
            An array with the same shape as `x_new` containing the true values
            for the predictions.
        """
        
        window = self._sliding_window.window
        stride = self._sliding_window.stride
        points = self._sliding_window.points_seen
        
        try:
            dimensions = self._sliding_window.forecast
            first_point = window
            dimensions_str = "forecast"
        except AttributeError:
            dimensions = self._sliding_window.window
            first_point = 0
            dimensions_str = "window"
        
        if y is None:
            raise ValueError("y must be an array-like with shape shape as x")
        
        x = np.array(x)
        y = np.array(y)
        
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        elif x.ndim != 3:
            raise ValueError(f"x must have 3 dimensions (n_windows, {dimensions_str}, features)")
        elif x.shape[1] != dimensions:
            raise ValueError(f"the input has wrong shape, x.shape[1] should be {dimensions}")
        
        point_pred = np.full((points, x.shape[2], dimensions), np.nan)
        point_true = np.full((points, x.shape[2], dimensions), np.nan)
        
        # iterate over all windows
        for i in range(x.shape[0]):
            first = first_point + i * stride
            last = first + dimensions
            
            # iterate over all features
            for f in range(x.shape[2]):
                np.fill_diagonal(point_pred[first:last, f, :], x[i, :, f])
                np.fill_diagonal(point_true[first:last, f, :], y[i, :, f])

        point_pred = point_pred.reshape(points, x.shape[2] * dimensions)
        point_true = point_true.reshape(points, y.shape[2] * dimensions)
        return point_pred, point_true
