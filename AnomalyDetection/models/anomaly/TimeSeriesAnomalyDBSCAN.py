# Python imports
from typing import Tuple

# External imports
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.utils import check_array

# Project imports
from input_validation.attribute_checks import check_attributes_exists
from models.transformers.TimeSeriesAnomalyLabeller import TimeSeriesAnomalyLabeller
from models.transformers.TimeSeriesAnomalyScorer import TimeSeriesAnomalyScorer
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyDBSCAN(BaseEstimator, OutlierMixin):
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
		
	window_scoring: {"z-score", "centroid"}, default="centroid"
		It defines the method with the point anomalies are computed. With
		"centroid" the anomaly is computed as euclidean distance from the
		closest centroid. Then, all the scores are normalized using min-max.
		With z-score, the anomalies are computed using the z-score.

	scaling: {"none", "minmax"}, default="minmax"
		The scaling method to scale the anomaly scores.

	scoring: {"average"}, default="average"
		The scoring method used compute the anomaly scores.
	
	classification: {"voting", "points_score"}, default="voting"
		It defines the way in which a point is declared as anomaly. With voting,
		a point is an anomaly if at least anomaly_threshold percentage of
		windows containing the point agree in saying it is an anomaly. With
		points_score, the points are considered anomalies if they're score is
		above anomaly_threshold.
	
	anomaly_threshold: float, default=None
		The threshold used to compute if a point is an anomaly or not. It will
		be passed to TimeSeriesAnomalyLabeller, see it for more details.

	anomaly_contamination: float, default=0.01
		The percentage of anomaly points in the dataset.
		
	Attributes
	----------
	labels_ : ndarray of shape (n_samples)
		Anomaly labels for points, 0 for normal points and 1 for anomalies.
		
	scores_ : ndarray of shape (n_samples)
		Anomaly scores for each point. The higher the score, the more likely the
		point is an anomaly.
	
	See Also
	--------
	For all the other parameters, see the scikit-learn implementation.
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
	"""
	WINDOW_SCORING_METHODS = ["z-score", "centroid"]
	CLASSIFICATIONS = ["voting", "points_score"]
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 window_scoring: str = "centroid",
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 anomaly_threshold: float = None,
				 anomaly_contamination: float = 0.01,
				 eps: float = 0.5,
				 min_samples: int = 5,
				 metric: str = "euclidean",
				 metric_params: dict = None,
				 algorithm: str = "auto",
				 leaf_size: int = 30,
				 p: float = None,
				 n_jobs: int = None):
		super().__init__()

		self.eps = eps
		self.min_samples = min_samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs

		self.dbscan: DBSCAN = None
		self.window = window
		self.stride = stride
		self.window_scoring = window_scoring
		self.classification = classification
		self.anomaly_threshold = anomaly_threshold
		self.scaling = scaling
		self.scoring = scoring
		self.anomaly_contamination = anomaly_contamination

		self._check_parameters()

	def set_params(self, **params):
		super().set_params(**params)
		self._check_parameters()

	def anomaly_score(self, X):
		"""Returns the anomaly score of the points.

		X : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		scores : ndarray of shape (n_samples,)
			The scores for the points between 0 and 1. The higher, the more
			abnormal. If threshold is not None, points above threshold are
			labelled as anomalies. Otherwise, see how points are labelled.
		"""
		check_attributes_exists(self, ["scores_", "labels_"])
		return self.scores_

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
		check_array(X)
		X = np.array(X)

		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Run vanilla dbscan on the vector space of the time series
		self._dbscan()
		self.dbscan.fit(X_new)
		clusters = set(self.dbscan.labels_).difference({-1})

		# Compute centroids to be able to compute the anomaly score
		centroids = []
		for cluster in clusters:
			cluster_points = np.argwhere(self.dbscan.labels_ == cluster)
			centroids.append(np.mean(X_new[cluster_points]))
		centroids = np.array(centroids)
		anomalies = np.argwhere(self.dbscan.labels_ == -1)
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
			scorer = TimeSeriesAnomalyScorer(self.window,
											 self.stride,
											 self.scaling,
											 self.scoring)
			labeller = TimeSeriesAnomalyLabeller(self.window,
												 self.stride,
												 self.anomaly_threshold,
												 self.anomaly_contamination,
												 self.classification)
			self.scores_ = scorer.fit_transform(window_scores,
												projector.num_windows_)
			self.labels_, _ = labeller.fit_transform(window_anomalies,
													 projector.num_windows_,
													 scores=self.scores_)

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
		
		if self.window_scoring == "z-score":
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

	def _check_parameters(self):
		"""Checks that the objects parameters are correct.

		Returns
		-------
		None
		"""
		if self.window_scoring not in self.WINDOW_SCORING_METHODS:
			raise ValueError("The score method must be one of",
							 self.WINDOW_SCORING_METHODS)
		elif self.classification not in self.CLASSIFICATIONS:
			raise ValueError("The classification must be one of",
							 self.CLASSIFICATIONS)

	def _dbscan(self) -> None:
		"""Instantiates the DBSCAN model as specified.

		Returns
		-------
		None
		"""
		self.dbscan = DBSCAN(self.eps,
							 min_samples=self.min_samples,
							 metric=self.metric,
							 metric_params=self.metric_params,
							 algorithm=self.algorithm,
							 leaf_size=self.leaf_size,
							 p=self.p,
							 n_jobs=self.n_jobs)
