# Python imports
import abc
from math import floor
from typing import Tuple

# External imports
import numpy as np

# Project imports
from models.anomaly.AnomalyLearner import AnomalyLearner


class AnomalyWindowUnsupervised(AnomalyLearner):
	"""WindowLearner"""
	ERROR_KEY = AnomalyLearner.ERROR_KEY.copy() + ["window_anomaly"]

	def __init__(self, window: int = 200,
				 stride: int = 200,
				 anomaly_threshold: float = 1.0):
		super().__init__()
		self._check_assumptions(window=window,
								stride=stride,
								anomaly_threshold=anomaly_threshold)
		
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold

	def _check_assumptions(self, *args,
						   **kwargs) -> None:
		super()._check_assumptions(args, kwargs)
		"""Checks if the assumption about the specified variable are true."""
		if ("window" in kwargs.keys()) ^ ("stride" in kwargs.keys()):
			raise ValueError(self._raise_error("window_need_stride"))
		else:
			window = kwargs["window"]
			stride = kwargs["stride"]
			if (window is not None) ^ (stride is not None):
				raise ValueError(self._raise_error("window_need_stride"))
			elif window is not None and stride > window:
				raise ValueError(self._raise_error("stride_lt_window"))

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
		"""
		# Check assumptions
		if train is None or train.shape[0] < 1:
			raise ValueError(self._raise_error("not_none"))
		elif train.ndim != 2:
			raise ValueError(self._raise_error("format"))
		elif train.shape[1] > 1:
			raise ValueError(self._raise_error("only_univariate"))

		# Set up values for stride and window
		if self.window is None:
			self.window = train.shape[0]
			self.stride = self.window

		# Preprocess data to add another dimension representing the index
		self.anomaly_scores = np.zeros(train.shape[0])
		self.anomalies = np.zeros(train.shape[0])
		num_evaluations = np.zeros(train.shape[0])
		
		for i in range(0, train.shape[0] - self.window, self.stride):
			must_include_more = train.shape[0] - i - self.window >= self.window
			if must_include_more:
				window_data = train[i:i + self.window]
			else:
				window_data = train[i:]
			
			window_scores = self._fit_window(window_data, labels, idx=i)
			self.anomaly_scores[i:i + len(window_data)] += window_scores
			num_evaluations[i:i + len(window_data)] += 1
		
		# Average all points that have been evaluated by more windows
		# NOTE: we can avoid storing num_evaluations and retrieve by the id the
		# number of time a point is evaluated and scored. However, the
		# complexity is around 30 lines. Therefore, I beg for storing the value.
		self.anomaly_scores = self.anomaly_scores / num_evaluations
		
		# Reshape the list onto a simple list
		self._compute_anomalies(num_evaluations)
	
	@abc.abstractmethod
	def _compute_anomalies(self, num_evaluations: np.ndarray,
						   *args,
						   **kwargs) -> None:
		"""Compute which are the samples categorized as anomaly."""
		pass
	
	@abc.abstractmethod
	def _fit_window(self, window_data: np.ndarray,
					labels: np.ndarray = None,
					*args,
					**kwargs) -> np.ndarray:
		"""Fit the dbscan to the window and return the computed anomalies.

		Parameters
		----------
		window_data : ndarray
			A window of data on which to perform anomalies search.

		Returns
		-------
		anomaly_scores : ndarray of shape window_data.shape[0]
			The anomaly scores for the points of the window dataset.
		"""
		pass
