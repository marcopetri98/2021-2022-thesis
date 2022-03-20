# Python imports
from typing import Tuple

# External imports
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.base import OutlierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

# Project imports
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyDBSCAN(DBSCAN, OutlierMixin):
	"""Concrete class representing the application of DBSCAN approach to time series.
	
	It is a wrapper of the scikit-learn DBSCAN approach. It uses the
	TimeSeriesProjector to project the time series onto a vector space. Then,
	it uses DBSCAN to find all the anomalies and compute the score of an anomaly
	as described in the fit_predict method. Please, note that the vanilla
	DBSCAN implementation does not produce anomaly scores.

	Parameters
	----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
		
	score_method: {"z-score", "centroid"}, default="centroid"
		It defines the method with the point anomalies are computed. With
		"centroid" the anomaly is computed as euclidean distance from the
		closest centroid. Then, all the scores are normalized using min-max.
		With z-score, the anomalies are computed using the z-score.
	
	classification: {"voting", "points_score"}, default="voting"
		It defines the way in which a point is declared as anomaly. With voting,
		a point is an anomaly if at least anomaly_threshold percentage of
		windows containing the point agree in saying it is an anomaly. With
		points_score, the points are considered anomalies if they're score is
		above anomaly_threshold.
	
	anomaly_threshold: float, default=0.0
		The threshold used to compute if a point is an anomaly or not.
		
	eps : float, default=0.5
		The maximum distance between two samples for one to be considered
		as in the neighborhood of the other. This is not a maximum bound
		on the distances of points within a cluster. This is the most
		important DBSCAN parameter to choose appropriately for your data set
		and distance function.

	min_samples : int, default=5
		The number of samples (or total weight) in a neighborhood for a point
		to be considered as a core point. This includes the point itself.

	metric : str, or callable, default='euclidean'
		The metric to use when calculating distance between instances in a
		feature array. If metric is a string or callable, it must be one of
		the options allowed by :func:`sklearn.metrics.pairwise_distances` for
		its metric parameter.
		If metric is "precomputed", X is assumed to be a distance matrix and
		must be square. X may be a :term:`Glossary <sparse graph>`, in which
		case only "nonzero" elements may be considered neighbors for DBSCAN.

		.. versionadded:: 0.17
		   metric *precomputed* to accept precomputed sparse matrix.

	metric_params : dict, default=None
		Additional keyword arguments for the metric function.

		.. versionadded:: 0.19

	algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
		The algorithm to be used by the NearestNeighbors module
		to compute pointwise distances and find nearest neighbors.
		See NearestNeighbors module documentation for details.

	leaf_size : int, default=30
		Leaf size passed to BallTree or cKDTree. This can affect the speed
		of the construction and query, as well as the memory required
		to store the tree. The optimal value depends
		on the nature of the problem.

	p : float, default=None
		The power of the Minkowski metric to be used to calculate distance
		between points. If None, then ``p=2`` (equivalent to the Euclidean
		distance).

	n_jobs : int, default=None
		The number of parallel jobs to run.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.
		
	Attributes
	----------
	labels_ : ndarray of shape (n_samples)
		Anomaly labels for points, 0 for normal points and 1 for anomalies.
		
	scores_ : ndarray of shape (n_samples)
		Anomaly scores for each point. The higher the score, the more likely the
		point is an anomaly.
	
	See Also
	--------
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
	"""
	SCORE_METHODS = ["z-score", "centroid"]
	CLASSIFICATIONS = ["voting", "points_score"]
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 score_method: str = "centroid",
				 classification: str = "voting",
				 anomaly_threshold: float = 0.0,
				 eps: float = 0.5,
				 min_samples: int = 5,
				 metric: str = "euclidean",
				 metric_params: dict = None,
				 algorithm: str = "auto",
				 leaf_size: int = 30,
				 p: float = None,
				 n_jobs: int = None):
		super().__init__(eps,
						 min_samples=min_samples,
						 metric=metric,
						 metric_params=metric_params,
						 algorithm=algorithm,
						 leaf_size=leaf_size,
						 p=p,
						 n_jobs=n_jobs)
		self.window = window
		self.stride = stride
		self.score_method = score_method
		self.classification = classification
		self.anomaly_threshold = anomaly_threshold
	
	def fit(self, X, y=None, sample_weight=None) -> None:
		"""Compute the anomalies on the time series.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.
		sample_weight : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		None
			Fits the model to the data.
		"""
		if self.score_method not in self.SCORE_METHODS:
			raise ValueError("The score method must be one of",
							 self.SCORE_METHODS)
		elif self.classification not in self.CLASSIFICATIONS:
			raise ValueError("The classification must be one of",
							 self.CLASSIFICATIONS)
		check_array(X)
		X = np.array(X)
		
		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)
		
		# Run vanilla dbscan on the vector space of the time series
		super().fit(X_new)
		clusters = set(self.labels_).difference({-1})
		
		# Compute centroids to be able to compute the anomaly score
		centroids = []
		for cluster in clusters:
			cluster_points = np.argwhere(self.labels_ == cluster)
			centroids.append(np.mean(X_new[cluster_points]))
		centroids = np.array(centroids)
		anomalies = np.argwhere(self.labels_ == -1)
		anomalies = anomalies.reshape(anomalies.shape[0])
		
		window_anomalies = np.zeros(X_new.shape[0])
		window_anomalies[anomalies] = 1
		window_scores = self._compute_window_scores(X_new, centroids)
		
		# Compute the anomaly labels and scores on the initial dataset
		if np.max(window_scores) == np.inf:
			# There is no centroid, all points are anomalies
			self.labels_ = np.ones(X.shape[0])
			self.scores_ = np.ones(X.shape[0])
		else:
			# There is at least one centroid, anomaly scores are computed
			# TODO: Identical to LOF code, find a way to avoid code duplication
			self.labels_ = np.zeros(X.shape[0])
			self.scores_ = np.zeros(X.shape[0])
			
			# Compute score of each point
			for i in range(window_scores.shape[0]):
				idx = i * self.stride
				self.scores_[idx:idx + self.window] += window_scores[i]
			self.scores_ = self.scores_ / projector.num_windows_
			
			if self.classification == "voting":
				# Anomalies are computed by voting of window anomalies
				for i in range(window_scores.shape[0]):
					if window_anomalies[i] == 1:
						idx = i * self.stride
						self.labels_[idx:idx + self.window] += 1
				self.labels_ = self.labels_ / projector.num_windows_
				
				true_anomalies = np.argwhere(self.labels_ > self.anomaly_threshold)
				self.labels_ = np.zeros(self.labels_.shape)
				self.labels_[true_anomalies] = 1
			else:
				self.labels_[np.argwhere(self.scores_ > self.anomaly_threshold)] = 1
			
			# Min-max normalization
			self.scores_ = self.scores_.reshape((self.scores_.shape[0], 1))
			self.scores_ = MinMaxScaler().fit_transform(self.scores_)
			self.scores_ = self.scores_.reshape(self.scores_.shape[0])
	
	def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.
		sample_weight : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		labels
			The labels for the points on the dataset.
		"""
		self.fit(X, y)
		return self.labels_

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
