from abc import ABC

import numpy as np
from sklearn.utils import check_array

from input_validation import check_array_general
from models import IMultipleParametric
from models.time_series.anomaly.machine_learning import TimeSeriesAnomalyWindowWrapper


class TSAMultipleParametric(TimeSeriesAnomalyWindowWrapper, IMultipleParametric, ABC):
    """A machine learning AD multiple parametric model."""
    
    def __init__(self, window: int = 5,
				 stride: int = 1,
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 threshold: float = None,
				 anomaly_portion: float = 0.01):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

    def fit(self, x, y=None, *args, **kwargs) -> None:
        check_array(x)
        x = np.array(x)
    
        x_new, windows_per_point = self._project_time_series(x)
        self._build_wrapped()
        self._wrapped_model.fit(x_new)

    def fit_multiple(self, x, y=None, *args, **kwargs) -> None:
        check_array_general(x, 3, (1, 1, 1))
        x = np.array(x)
    
        x_total = None
        x_new = list()
        windows_per_point = list()
        for series in x:
            ser_new, ser_windows_per_point = self._project_time_series(series)
            x_new.append(ser_new)
            windows_per_point.append(ser_windows_per_point)
        
            if x_total is None:
                x_total = x_new
            else:
                x_total = np.append(x_total, x_new)
    
        self._build_wrapped()
        self._wrapped_model.fit(x_new)
