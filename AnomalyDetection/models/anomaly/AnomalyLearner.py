# Python imports
from abc import ABC

# External imports
import numpy as np

# Project imports
from base.BaseObject import mix_keys
from models.SupervisedLearner import SupervisedLearner
from models.UnsupervisedLearner import UnsupervisedLearner


class AnomalyLearner(UnsupervisedLearner, SupervisedLearner, ABC):
	"""Abstract class used to define an anomaly learner"""
	ERROR_KEY = mix_keys(UnsupervisedLearner, SupervisedLearner) + ["anomaly"]
	
	def __init__(self):
		super().__init__()
		self.anomaly_scores = None
		self.anomalies = None
	
	def get_anomaly_scores(self, *args,
						   **kwargs) -> np.ndarray:
		"""Gets the anomaly scores.

		Returns
		-------
		anomaly_scores: ndarray
			The anomaly scores of the points of the dataset.
		"""
		if self.anomaly_scores is None:
			raise ValueError(self._raise_error("fit_before"))
		
		return self.anomaly_scores.copy()
	
	def get_anomalies(self, *args,
					  **kwargs) -> np.ndarray:
		"""Gets the anomalies position.

		Returns
		-------
		anomaly_scores: ndarray
			The anomaly scores of the points of the dataset.
		"""
		if self.anomalies is None:
			raise ValueError(self._raise_error("fit_before"))
		
		return self.anomalies.copy()