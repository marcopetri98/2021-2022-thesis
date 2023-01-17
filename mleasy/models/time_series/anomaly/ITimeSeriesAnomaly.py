from abc import ABC

from ...IAnomalyClassifier import IAnomalyClassifier
from ...IAnomalyRegressor import IAnomalyRegressor


class ITimeSeriesAnomaly(IAnomalyRegressor, IAnomalyClassifier, ABC):
    """Interface identifying a machine learning time series anomaly detector.
    """
