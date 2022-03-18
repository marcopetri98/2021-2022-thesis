# Python imports
from typing import Tuple

# External imports
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, OutlierMixin

# Project imports


class TimeSeriesAnomalyDBSCAN(BaseEstimator, OutlierMixin):
	"""Concrete class representing the application of DBSCAN approach to time series.
	
	Attributes
	----------
	window : int
		The length of the window to consider performing anomaly detection.
	stride : int
		The offset at which the window is moved when computing the anomalies.
	"""
	VALID_SCORE_METHODS = ["z-score", "centroid"]
	
	def __init__(self, eps: float = 0.5,
				 min_points: int = 5,
				 metric: str = "euclidean",
				 metric_params: dict = None,
				 algorithm: str = "auto",
				 leaf_size: int = 30,
				 p: float = None,
				 n_jobs: int = None,
				 window: int = None,
				 stride: int = None,
				 anomaly_threshold: float = 0.5,
				 score_method: str = "centroid",
				 classification: str = "auto"):
		super().__init__(window=window,
						 stride=stride,
						 anomaly_threshold=anomaly_threshold,
						 classification=classification)
		self._check_assumptions(window=window,
								stride=stride,
								score_method=score_method)
		
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
		self.score_method = score_method
	
	def _check_assumptions(self, *args,
						   **kwargs) -> None:
		"""Checks if the assumption about the specified variable are true."""
		super()._check_assumptions(args, kwargs)
		if "score_method" in kwargs.keys():
			score_method = kwargs["score_method"]
			if score_method not in self.VALID_SCORE_METHODS:
				raise ValueError(self._raise_error("valid_anomaly_metrics"))

	def _compute_anomaly_scores(self, num_evaluations: np.ndarray,
								*args,
								**kwargs) -> None:
		if np.max(self._windowed_scores) == np.inf:
			# All points are labelled as anomalies
			self.anomaly_scores = np.ones(num_evaluations.shape[0])
		else:
			# There is at least one cluster
			super()._compute_anomaly_scores(num_evaluations)
			self.anomaly_scores = self.anomaly_scores / np.max(self.anomaly_scores)

	def _compute_anomalies(self, num_evaluations: np.ndarray,
						   *args,
						   **kwargs) -> None:
		"""Compute which are the samples categorized as anomaly."""
		if np.max(self._windowed_scores) == np.inf:
			# All points are labelled as anomalies
			self.anomalies = np.ones(num_evaluations.shape[0])
		else:
			super()._compute_anomalies(num_evaluations)

	def _fit_windowed_data(self, spatial_data: np.ndarray,
						   labels: np.ndarray = None,
						   num_points: int = 0,
						   *args,
						   **kwargs) -> None:
		"""Fit the dbscan to the window and return the computed anomalies.

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
		anomaly_scores : ndarray of shape window_data.shape[0]
			The anomaly scores for the points of the window dataset.
		"""
		dbscan = DBSCAN(self.eps,
						min_samples=self.min_points,
						metric=self.metric,
						metric_params=self.metric_params,
						algorithm=self.algorithm,
						leaf_size=self.leaf_size,
						p=self.p,
						n_jobs=self.n_jobs)
		dbscan.fit(spatial_data)
		cluster_labels = dbscan.labels_
		clusters = set(cluster_labels).difference({-1})
		
		centroids = []
		for cluster in clusters:
			cluster_points = np.argwhere(cluster_labels == cluster)
			centroids.append(np.mean(spatial_data[cluster_points]))
		
		centroids = np.array(centroids)
		anomalies = np.argwhere(cluster_labels == -1)
		anomalies = anomalies.reshape(anomalies.shape[0])
		
		self._windowed_anomalies = np.zeros(spatial_data.shape[0])
		self._windowed_anomalies[anomalies] = 1
		self._windowed_scores = self._compute_window_scores(spatial_data,
															centroids)

	def _compute_window_scores(self, spatial_data: np.ndarray,
								centroids: np.array,
								*args,
								**kwargs) -> np.ndarray:
		"""Compute the anomaly scores of the data.
		
		Parameters
		----------
		spatial_data : ndarray
			A window of data on which to perform anomalies search.

		Returns
		-------
		anomaly_scores : ndarray with shape data.shape
			An array with the anomaly scores for each point in the time series,
			computed as the z-score.
		"""
		anomaly_scores = np.zeros(spatial_data.shape[0])
		
		if self.score_method == "z-score":
			mean = np.average(spatial_data, axis=0)
			std = np.std(spatial_data, axis=0, ddof=1)
			
			# Compute the anomaly scores using z-score
			for i in range(spatial_data.shape[0]):
				deviated_point = (spatial_data[i] - mean) / std
				anomaly_scores[i] = np.linalg.norm(deviated_point)
		else:
			# Compute the anomaly scores using distance from the closest centroid
			for i in range(spatial_data.shape[0]):
				min_distance = np.inf
				
				for j in range(centroids.shape[0]):
					distance = np.linalg.norm(spatial_data[i] - centroids[j])
					if distance < min_distance:
						min_distance = distance
				
				anomaly_scores[i] = min_distance
			
		return anomaly_scores
