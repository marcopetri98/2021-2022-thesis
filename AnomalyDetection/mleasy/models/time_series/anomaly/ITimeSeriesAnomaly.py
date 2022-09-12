from abc import ABC

from mleasy.models.IAnomalyClassifier import IAnomalyClassifier
from mleasy.models.IAnomalyRegressor import IAnomalyRegressor


class ITimeSeriesAnomaly(IAnomalyRegressor, IAnomalyClassifier, ABC):
    """Interface identifying a machine learning time series anomaly detector.
    """
