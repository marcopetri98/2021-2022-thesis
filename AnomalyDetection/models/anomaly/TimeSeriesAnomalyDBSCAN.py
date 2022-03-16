# Python imports
from typing import Tuple

# External imports
import numpy as np
import sklearn.cluster as sk

# Project imports
from models.anomaly.AnomalyLearner import AnomalyLearner


class TimeSeriesAnomalyDBSCAN(AnomalyLearner):
	"""Concrete class representing the application of DBSCAN approach to time series."""
	ERROR_KEY = AnomalyLearner.ERROR_KEY.copy() + ["dbscan"]
	
	def __init__(self, eps: float = 0.5,
				 min_points: int = 5,
				 metric: str = "euclidean",
				 metric_params: dict = None,
				 algorithm: str = "auto",
				 leaf_size: int = 30,
				 p: float = None,
				 n_jobs: int = None,
				 window: int = None,
				 stride: int = None):
		super().__init__()
		self.check_assumptions({"window": window, "stride": stride})
		
		self.eps = eps
		self.min_points = min_points
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs
		self.window = window
		self.stride = stride
		self.clusters = None
		self.centroids = None
	
	def check_assumptions(self, *args,
						  **kwargs) -> None:
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
	
	# TODO: implement stride different from window to implement robustness
	def fit(self, data: np.ndarray,
			*args,
			**kwargs) -> None:
		"""Fit the dbscan model to the time series data using scikit-learn.
		
		Parameters
		----------
		data : ndarray of shape (n_samples, n_features)
			The time series data without containing the index, timestmap or not.
		window : int
			The length of the window to consider performing anomaly detection.
		stride : int
			The offset at which the window is moved when computing the anomalies.
		
		Returns
		-------
		None
		
		"""
		# Check assumptions
		if data is None or data.shape[0] < 1:
			raise ValueError(self._raise_error("not_none"))
		elif data.ndim != 2:
			raise ValueError(self._raise_error("format"))
		elif data.shape[1] > 1:
			raise ValueError(self._raise_error("only_univariate"))
		
		# Set up values for stride and window
		if self.window is None:
			self.window = data.shape[0]
			self.stride = self.window
		
		# Preprocess data to add another dimension representing the index
		self.anomaly_scores = np.zeros(data.shape[0])
		self.anomalies = np.zeros(data.shape[0])
		anomalies_idx = []
		
		for i in range(0, data.shape[0] - self.window, self.stride):
			must_include_more = data.shape[0] - i - self.window >= self.window
			if must_include_more:
				window_data = data[i:i + self.window]
				self.anomaly_scores[i:i + self.window], anomalies = self._fit_window(window_data)
			else:
				window_data = data[i:]
				self.anomaly_scores[i:], anomalies = self._fit_window(window_data)
			
			if len(anomalies) > 0:
				anomalies = anomalies + i
				anomalies_idx.append(anomalies.tolist())

		# Reshape the list onto a simple list
		anomalies_idx = [x for z in anomalies_idx for y in z for x in y]
		self.anomalies[np.array(anomalies_idx, dtype=np.intc)] = 1

	def _fit_window(self, window_data: np.ndarray,
					*args,
					**kwargs) -> Tuple[np.ndarray, np.ndarray]:
		"""Fit the dbscan to the window and return the computed anomalies.
		
		Parameters
		----------
		window_data : ndarray
			A window of data on which to perform anomalies search.
		
		Returns
		-------
		anomaly_scores : ndarray of shape window_data.shape[0]
			The anomaly scores for the points of the window dataset.
		anomalies : ndarray of shape window_data.shape[0]
			The indices of the anomalies in this window.
		"""
		dbscan = sk.DBSCAN(self.eps,
						   min_samples=self.min_points,
						   metric=self.metric,
						   metric_params=self.metric_params,
						   algorithm=self.algorithm,
						   leaf_size=self.leaf_size,
						   p=self.p,
						   n_jobs=self.n_jobs)
		dbscan.fit(window_data)
		cluster_labels = dbscan.labels_
		
		# Compute the centroids to be able to compute anomaly score
		# FIXME: AM I USEFUL?
		centroids = []
		are_there_anomalies = True if -1 in cluster_labels else False
		num_clusters = len(np.unique(cluster_labels)) - are_there_anomalies
		for i in range(num_clusters):
			# Mask points for the current label
			num_points = np.sum(cluster_labels == i)
			current_label = cluster_labels == i
			points = np.extract(current_label, window_data)
			points.reshape((num_points, window_data.shape[1]))
			
			# Compute the centroid
			centroids.append(np.average(window_data, axis=0))
		
		anomalies = np.argwhere(cluster_labels == -1)
		scores = self._compute_anomaly_scores(window_data)
		return scores, anomalies

	@staticmethod
	def _compute_anomaly_scores(window_data: np.ndarray,
								*args,
								**kwargs) -> np.ndarray:
		"""Compute the anomaly scores of the data.
		
		Parameters
		----------
		window_data : ndarray
			A window of data on which to perform anomalies search.

		Returns
		-------
		anomaly_scores : ndarray with shape data.shape
			An array with the anomaly scores for each point in the time series,
			computed as the z-score.
		"""
		mean = np.average(window_data, axis=0)
		std = np.std(window_data, axis=0, ddof=1)
		anomaly_scores = np.zeros(window_data.shape[0])
		
		# Compute the anomaly scores using z-score
		for i in range(window_data.shape[0]):
			anomaly_scores[i] = (window_data[i] - mean) / std
			anomaly_scores[i] = np.linalg.norm(anomaly_scores[i])
			
		return anomaly_scores
