# Python imports
from typing import Union, Callable

# External imports
import numpy as np
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_array

# Project imports
from sklearn.utils.validation import check_is_fitted

from models.anomaly.IAnomaly import IAnomaly
from models.transformers.TimeSeriesAnomalyLabeller import TimeSeriesAnomalyLabeller
from models.transformers.TimeSeriesAnomalyScorer import TimeSeriesAnomalyScorer
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyLOF(BaseEstimator, OutlierMixin, IAnomaly):
	"""LOF adapter for time series.

	It is a wrapper of the scikit-learn LOF approach. It uses the
	TimeSeriesProjector to project the time series onto a vector space. Then,
	it uses LOF to find all the anomalies and compute the score of an anomaly
	as described in the fit_predict method.
	
	Parameters
	----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
	
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
	https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
	"""
	CLASSIFICATIONS = TimeSeriesAnomalyLabeller.ACCEPTED_LABELLING_METHODS
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 classification: str = "voting",
				 scaling: str = "minmax",
				 scoring: str = "average",
				 anomaly_threshold: float = None,
				 anomaly_contamination: float = 0.01,
				 n_neighbors: int = 20,
				 algorithm: str = 'auto',
				 leaf_size: int = 30,
				 metric: Union[str, Callable[[list, list], float]] = 'minkowski',
				 p: int = 2,
				 metric_params: dict = None,
				 contamination: Union[str, float] = 'auto',
				 novelty: bool = False,
				 n_jobs: int = None):
		super().__init__()

		self.n_neighbors = n_neighbors
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.contamination = contamination
		self.novelty = novelty
		self.n_jobs = n_jobs

		self.lof: LocalOutlierFactor = None
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold
		self.classification = classification
		self.scaling = scaling
		self.scoring = scoring
		self.anomaly_contamination = anomaly_contamination

		self._check_parameters()

	def set_params(self, **params):
		super().set_params(**params)
		self._check_parameters()

	def anomaly_score(self, X):
		"""Returns the anomaly score of the points.

		X : array-like of shape (n_samples, n_features)
			Data for which we want to predict whether the points are normal or
			abnormal.

		Returns
		-------
		scores : ndarray of shape (n_samples,)
			The scores for the points between 0 and 1. The higher, the more
			abnormal. If threshold is not None, points above threshold are
			labelled as anomalies. Otherwise, see how points are labelled.
		"""
		check_array(X)
		X = np.array(X)

		if self.lof.novelty:
			# If the method is used as novelty detection, scores are computed
			# Project time series onto vector space
			projector = TimeSeriesProjector(self.window, self.stride)
			X_new = projector.fit_transform(X)

			# Predict the if the given points are abnormal
			window_scores = self.lof.decision_function(X_new)
			window_scores = window_scores * -1

			# Compute scores and labels for points
			point_scores = self._compute_scores(window_scores,
												projector.num_windows_)
		else:
			# If the method is used as unsupervised, labels are already there
			check_is_fitted(self, ["scores_", "labels_"])
			point_scores = self.scores_

		return point_scores

	def fit(self, X, y=None):
		"""Fits the model with the given training data points.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data. Note
			that these data must be composed of only normal data and not of
			abnormal data.
		y : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		None
			Fitted instance of itself.
		"""
		X = np.array(X)

		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Run vanilla LOF on the vector space of the time series
		self._lof()
		self.lof.fit(X_new)

	def predict(self, X=None):
		"""Computes the prediction for the given data.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data for which we want to predict whether the points are normal or
			abnormal.

		Returns
		-------
		labels : ndarray of shape (n_samples,)
			The labels for the points on the dataset.
		"""
		if not self.novelty:
			raise ValueError("fit_predict is available only when novelty is"
							 " False")
		check_array(X)
		X = np.array(X)

		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Predict the if the given points are abnormal
		window_scores = self.lof.decision_function(X_new)
		anomalies = np.argwhere(window_scores < 0)
		window_anomalies = np.zeros(X_new.shape[0])
		window_anomalies[anomalies] = 1
		window_scores = window_scores * -1

		# Compute scores and labels for points
		point_scores = self._compute_scores(window_scores,
											projector.num_windows_)
		point_labels = self._compute_labels(window_anomalies,
											   projector.num_windows_,
											   point_scores)

		return point_labels

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
		labels : ndarray of shape (n_samples,)
			The labels for the points on the dataset.
		"""
		if self.novelty:
			raise ValueError("fit_predict is available only when novelty is"
							 " False")
		check_array(X)
		X = np.array(X)

		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Run vanilla LOF on the vector space of the time series
		self._lof()
		self.lof.fit_predict(X_new)

		anomalies = np.argwhere(self.lof.negative_outlier_factor_ < self.lof.offset_)
		window_anomalies = np.zeros(X_new.shape[0])
		window_anomalies[anomalies] = 1
		window_scores = - self.lof.negative_outlier_factor_

		# Compute scores and labels for points
		self.scores_ = self._compute_scores(window_scores,
											projector.num_windows_)
		self.labels_ = self._compute_labels(window_anomalies,
											   projector.num_windows_,
											   self.scores_)
		return self.labels_

	def _compute_scores(self, window_scores: np.ndarray,
						num_windows_per_point: np.ndarray) -> np.ndarray:
		"""Compute the score for each point.

		Parameters
		----------
		window_scores : ndarray of shape (n_windows,)
			The score of each window.

		num_windows_per_point : ndarray of shape (n_samples,)
			Numer of windows containing the point.

		Returns
		-------
		scores : ndarray of shape (n_samples,)
			The scores of each point.
		"""
		scorer = TimeSeriesAnomalyScorer(self.window,
										 self.stride,
										 self.scaling,
										 self.scoring)
		return scorer.fit_transform(window_scores,
									num_windows_per_point)

	def _compute_labels(self, window_anomalies: np.ndarray,
						num_windows_per_point: np.ndarray,
						scores: np.ndarray) -> np.ndarray:
		"""Compute the label for each point.

		Parameters
		----------
		window_anomalies : ndarray of shape (n_windows,)
			The label of each window.

		num_windows_per_point : ndarray of shape (n_samples,)
			Numer of windows containing the point.

		scores : ndarray of shape (n_samples,)
			The scores of each point.

		Returns
		-------
		labels : ndarray of shape (n_samples,)
			Labels for the points.
		"""
		labeller = TimeSeriesAnomalyLabeller(self.window,
											 self.stride,
											 self.anomaly_threshold,
											 self.anomaly_contamination,
											 self.classification)
		labels, _ = labeller.fit_transform(window_anomalies,
										   num_windows_per_point,
										   scores=scores)
		return labels

	def _check_parameters(self) -> None:
		"""Checks that the objects parameters are correct.

		Returns
		-------
		None
		"""
		if self.classification not in self.CLASSIFICATIONS:
			raise ValueError("The classification must be one of",
								 self.CLASSIFICATIONS)

	def _lof(self) -> None:
		"""Instantiates the LocalOutlierFactor model as specified.

		Returns
		-------
		None
		"""
		self.lof = LocalOutlierFactor(self.n_neighbors,
									  algorithm=self.algorithm,
									  leaf_size=self.leaf_size,
									  metric=self.metric,
									  p=self.p,
									  metric_params=self.metric_params,
									  contamination=self.contamination,
									  novelty=self.novelty,
									  n_jobs=self.n_jobs)
