# Python imports

# External imports
import numpy as np

# Project imports
from models.anomaly.AnomalyLearner import AnomalyLearner


class TimeSeriesLOF(AnomalyLearner):
	"""Concrete class representing the application of LOF approach to time series."""
	ERROR_KEY = AnomalyLearner.ERROR_KEY.copy() + ["lof"]
	
	def __init__(self):
		super().__init__()
	
	def fit(self, data: np.ndarray,
			labels: np.ndarray = None,
			*args,
			**kwargs) -> None:
		pass