# Python imports
from abc import ABC
from typing import Tuple

# External imports
import numpy as np

# Project imports
from base.BaseObject import mix_keys
from models.Learner import Learner


class AnomalyLearner(Learner, ABC):
	"""Abstract class used to define an anomaly learner"""
	
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
	
def project_time_series(window: int,
						stride: int,
						train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""From a univariate time series compute a vector dataset from window and stride.
	
	Parameters
	----------
	window: int
		The number of points composing a vector in the new vector space.
	stride: int
		The number of steps that must be performed on the time series before
		building the new point.
	train: ndarray
		The training dataset to translate into a vector space.
		
	Returns
	-------
	spatial_data: ndarray
		The data projected onto a vector space.
	num_evaluations: ndarray of shape (n_samples,)
		The number of windows having the point as an element.
	"""
	# Check assumptions
	if train is None or train.shape[0] < 1:
		raise ValueError("Data cannot be None and data must be at least composed of one point.")
	elif window > train.shape[0]:
		raise ValueError("Window cannot be larger than data size.")
	elif train.ndim != 2:
		raise ValueError("Only univariate time series is currently supported.")
	elif train.shape[1] > 1:
		raise ValueError("Only univariate time series is currently supported.")
	elif (train.shape[0] - window) % stride != 0:
		raise ValueError("Data.shape[0] - window must be a multiple of stride to build the spatial data.")
	
	# Number of times a point is considered in a window
	num_evaluations = np.zeros(train.shape[0])
	spatial_time_series = []
	
	# Transform univariate time series into spatial data
	for i in range(0, train.shape[0] - window + 1, stride):
		num_evaluations[i:i + window] += 1
		current_data: np.ndarray = train[i:i + window]
		current_data = current_data.reshape(current_data.shape[0])
		spatial_time_series.append(current_data.tolist())
	
	spatial_time_series = np.array(spatial_time_series)
	
	return spatial_time_series, num_evaluations
