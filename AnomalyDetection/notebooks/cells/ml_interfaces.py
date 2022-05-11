import abc
from abc import ABC

import numpy as np

class IClassifier(ABC):
    @abc.abstractmethod
    def classify(self, X, *args, **kwargs) -> np.ndarray:
        pass

class IRegressor(ABC):
    @abc.abstractmethod
    def regress(self, x, *args, **kwargs) -> np.ndarray:
        pass

class IParametric(ABC):
    @abc.abstractmethod
    def fit(self, x, y=None, *args, **kwargs) -> None:
        pass
    
class IAnomalyRegressor(IRegressor):
    @abc.abstractmethod
    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        pass
    
class IAnomalyClassifier(IClassifier, ABC):
    pass

class ITimeSeriesAnomaly(IAnomalyRegressor, IAnomalyClassifier, ABC):
    pass

class ITimeSeriesAnomalyWindow(ITimeSeriesAnomaly):
    @abc.abstractmethod
    def _project_time_series(self, time_series: np.ndarray) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _compute_point_scores(self, window_scores,
                              windows_per_point) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _compute_point_labels(self, window_labels,
                              windows_per_point,
                              point_scores=None) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        pass

class ITimeSeriesAnomalyWrapper(ABC):
    @abc.abstractmethod
    def _build_wrapped(self) -> None:
        pass