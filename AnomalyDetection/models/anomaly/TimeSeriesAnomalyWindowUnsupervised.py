# Python imports
import abc
from math import floor
from typing import Tuple

# External imports
import numpy as np

# Project imports
from models.anomaly.AnomalyLearner import AnomalyLearner, \
	project_time_series


class TimeSeriesAnomalyWindowUnsupervised(AnomalyLearner):
	"""Anomaly detection learner of time series with classical machine learning.
	
	An object inheriting this class must be a classical machine learning
	unsupervised approach aiming to use a sliding window approach for univariate
	time series. It projects univariate time series data onto spatial data by
	transposing univariate time series data in a window-dimensional space.
	"""
	CLASSIFICATIONS = ["auto", "point_threshold"]

	def __init__(self, window: int = 10,
				 stride: int = 1,
				 anomaly_threshold: float = 0.5,
				 classification: str = "auto"):
		super().__init__()
		
		self.window = window
		self.stride = stride
		self.classification = classification
		self.anomaly_threshold = anomaly_threshold
		self._windowed_anomalies = None
		self._windowed_scores = None
		
		self._check_assumptions(window=window,
								stride=stride,
								anomaly_threshold=anomaly_threshold,
								classification=classification)

	def _check_assumptions(self, *args,
						   **kwargs) -> None:
		super()._check_assumptions(args, kwargs)
		"""Checks if the assumption about the specified variable are true."""
		if "window" in kwargs.keys() or "stride" in kwargs.keys():
			if "window" in kwargs.keys() and "stride" in kwargs.keys():
				window = kwargs["window"]
				stride = kwargs["stride"]
			elif "window" in kwargs.keys() and "stride" not in kwargs.keys():
				window = kwargs["window"]
				stride = self.stride
			else:
				window = self.window
				stride = kwargs["stride"]
				
			if window is None or stride is None:
				raise ValueError(self._raise_error("window_stride_not_none"))
			elif window is not None and stride > window:
				raise ValueError(self._raise_error("stride_lt_window"))
		if "classification" in kwargs.keys():
			classification = kwargs["classification"]
			if classification not in self.CLASSIFICATIONS:
				raise ValueError(self._raise_error("classification_type"))

	def fit(self, train: np.ndarray,
			labels: np.ndarray = None,
			*args,
			**kwargs) -> None:
		"""Fit the LOF model to the time series data using scikit-learn.

		Parameters
		----------
		train : ndarray of shape (n_samples, n_features)
			The time series data without containing the index, timestmap or not.
			

		Returns
		-------
		None
			Fitted model with the estimated anomalies.
		"""
		spatial_time_series, num_evaluations = project_time_series(self.window,
																   self.stride,
																   train)
		# Average all points that have been evaluated by more windows
		# NOTE: we can avoid storing num_evaluations and retrieve by the id the
		# number of time a point is evaluated and scored. However, the
		# complexity is around 30 lines. Therefore, I beg for storing the value.
		self._fit_windowed_data(spatial_time_series,
								labels,
								train.shape[0])
		
		# Reshape the list onto a simple list
		self._compute_anomaly_scores(num_evaluations)
		self._compute_anomalies(num_evaluations)
	
	def _compute_anomaly_scores(self, num_evaluations: np.ndarray,
								*args,
								**kwargs) -> None:
		"""Compute the anomaly score of each sample by average of windows."""
		self.anomaly_scores = np.zeros(num_evaluations.shape[0])
			
		for i in range(self._windowed_scores.shape[0]):
			idx = i * self.stride
			self.anomaly_scores[idx:idx + self.window] += self._windowed_scores[i]
			
		self.anomaly_scores = self.anomaly_scores / num_evaluations
	
	def _compute_anomalies(self, num_evaluations: np.ndarray,
						   *args,
						   **kwargs) -> None:
		"""Compute which are the samples categorized as anomaly.
		
		If classification is auto, the points are classified as anomaly by
		voting with the specified percentage threshold. The threshold specifies
		the percentage of windows that agree that a point is an anomaly. If
		classification is point_threshold.
		"""
		self.anomalies = np.zeros(num_evaluations.shape[0])
		
		if self.classification == "auto":
			# Anomalies are computed by voting of window anomalies
			for i in range(self._windowed_scores.shape[0]):
				if self._windowed_anomalies[i] == 1:
					idx = i * self.stride
					self.anomalies[idx:idx + self.window] += 1
			self.anomalies = self.anomalies / num_evaluations
			
			true_anomalies = np.argwhere(self.anomalies > self.anomaly_threshold)
			self.anomalies = np.zeros(self.anomalies.shape)
			self.anomalies[true_anomalies] = 1
		else:
			self.anomalies[np.argwhere(self.anomaly_scores > self.anomaly_threshold)] = 1
	
	@abc.abstractmethod
	def _fit_windowed_data(self, spatial_data: np.ndarray,
						   labels: np.ndarray = None,
						   num_points: int = 0,
						   *args,
						   **kwargs) -> None:
		"""Fit the model to the window and return the computed anomalies.
		
		Parameters
		----------
		spatial_data : ndarray
			A window of data on which to perform anomalies search.
		labels: ndarray
			Labels for the points of the dataset.
		num_points: int
			Number of points in the original dataset.
		
		Returns
		-------
		None
			The fitted model using the windowed spatial data.
		"""
		pass
