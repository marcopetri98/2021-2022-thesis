from abc import ABC

from models.IAnomalyClassifier import IAnomalyClassifier
from models.IAnomalyRegressor import IAnomalyRegressor


class ITimeSeriesAnomaly(IAnomalyRegressor, IAnomalyClassifier, ABC):
	"""Interface identifying a machine learning time series anomaly detector.
    """
