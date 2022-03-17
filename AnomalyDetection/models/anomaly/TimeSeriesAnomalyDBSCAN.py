# Python imports
from typing import Tuple

# External imports
import numpy as np
import sklearn.cluster as sk

# Project imports
from models.anomaly.AnomalyWindowUnsupervised import AnomalyWindowUnsupervised


class TimeSeriesAnomalyDBSCAN(AnomalyWindowUnsupervised):
	"""Concrete class representing the application of DBSCAN approach to time series.
	
	Attributes
	----------
	window : int
		The length of the window to consider performing anomaly detection.
	stride : int
		The offset at which the window is moved when computing the anomalies.
	"""
	ERROR_KEY = AnomalyWindowUnsupervised.ERROR_KEY.copy() + ["dbscan"]
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
				 anomaly_threshold: float = 1.0,
				 use_score: bool = False,
				 score_method: str = "centroid"):
		super().__init__(window=window,
						 stride=stride,
						 anomaly_threshold=anomaly_threshold)
		self._check_assumptions(window=window,
								stride=stride,
								score_method=score_method,
								anomaly_threshold=anomaly_threshold)
		
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
		self.use_score = use_score
	
	def _check_assumptions(self, *args,
						   **kwargs) -> None:
		super()._check_assumptions(args, kwargs)
		"""Checks if the assumption about the specified variable are true."""
		score_method = kwargs["score_method"]
		if score_method not in self.VALID_SCORE_METHODS:
			raise ValueError(self._raise_error("valid_anomaly_metrics"))
	
	def fit(self, train: np.ndarray,
			*args,
			**kwargs) -> None:
		"""Fit the dbscan model to the time series data using scikit-learn.
		
		Parameters
		----------
		train : ndarray of shape (n_samples, n_features)
			The time series data without containing the index, timestmap or not.
		
		Returns
		-------
		None
		"""
		super().fit(train)

	def _compute_anomalies(self, num_evaluations: np.ndarray,
						   *args,
						   **kwargs) -> None:
		"""Compute which are the samples categorized as anomaly."""
		self.anomaly_scores = self.anomaly_scores / np.max(self.anomaly_scores)
		
		if not self.use_score:
			self.anomalies = self.anomalies / num_evaluations
			
			true_anomalies = np.argwhere(self.anomalies > self.anomaly_threshold)
			self.anomalies = np.zeros(self.anomalies.shape)
			self.anomalies[true_anomalies] = 1
		else:
			anomalies = np.argwhere(self.anomaly_scores > self.anomaly_threshold)
			self.anomalies = np.zeros(self.anomalies.shape)
			self.anomalies[anomalies] = 1

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
		clusters = set(cluster_labels).difference({-1})
		
		centroids = []
		for cluster in clusters:
			cluster_points = np.argwhere(cluster_labels == cluster)
			centroids.append(np.mean(window_data[cluster_points]))
		
		# TODO: implement multivariate
		anomalies = np.argwhere(cluster_labels == -1)
		anomalies = anomalies.reshape(anomalies.shape[0])
		self.anomalies[kwargs["idx"]:kwargs["idx"] + anomalies.shape[0]] += 1
		
		scores = self._compute_anomaly_scores(window_data, np.array(centroids))
		return scores

	def _compute_anomaly_scores(self, window_data: np.ndarray,
								centroids: np.array,
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
		anomaly_scores = np.zeros(window_data.shape[0])
		
		if self.score_method == "z-score":
			mean = np.average(window_data, axis=0)
			std = np.std(window_data, axis=0, ddof=1)
			
			# Compute the anomaly scores using z-score
			for i in range(window_data.shape[0]):
				anomaly_scores[i] = (window_data[i] - mean) / std
				anomaly_scores[i] = np.linalg.norm(anomaly_scores[i])
		else:
			# Compute the anomaly scores using distance from the closest centroid
			for i in range(window_data.shape[0]):
				min_distance = np.inf
				
				for j in range(centroids.shape[0]):
					distance = np.linalg.norm(window_data[i] - centroids[j])
					if distance < min_distance:
						min_distance = distance
				
				anomaly_scores[i] = min_distance
			
		return anomaly_scores
