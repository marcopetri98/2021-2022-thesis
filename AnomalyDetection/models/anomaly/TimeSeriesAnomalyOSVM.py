# Python imports

# External imports
import numpy as np
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array, check_X_y

# Project imports
from models.transformers.TimeSeriesAnomalyLabeller import \
	TimeSeriesAnomalyLabeller
from models.transformers.TimeSeriesAnomalyScorer import TimeSeriesAnomalyScorer
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyOSVM(BaseEstimator, OutlierMixin):
	"""OSVM adaptation to time series using scikit-learn.

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
        
    See Also
    --------
	For all the other parameters, see the scikit-learn implementation.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 anomaly_threshold: float = None,
				 anomaly_contamination: float = 0.01,
				 kernel: str = "rbf",
				 degree: int = 3,
				 gamma: str = "scale",
				 coef0: float = 0.0,
				 tol: float = 1e-3,
				 nu: float = 0.5,
				 shrinking: bool = True,
				 cache_size: float = 200,
				 verbose: bool = False,
				 max_iter: int = -1):
		super().__init__()

		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.tol = tol
		self.nu = nu
		self.shrinking = shrinking
		self.cache_size = cache_size
		self.verbose = verbose
		self.max_iter = max_iter

		self.osvm: OneClassSVM = None
		self.window = window
		self.stride = stride
		self.classification = classification
		self.anomaly_threshold = anomaly_threshold
		self.scaling = scaling
		self.scoring = scoring
		self.anomaly_contamination = anomaly_contamination
	
	def fit(self, X, y=None, sample_weight=None, **params) -> None:
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : array-like of shape (n_samples, n_features), default=None
			If given, the training will be performed only on samples from the
			class with label 0 and each value must be either 0 or 1. If None,
			all the points in X are used as training.
		sample_weight : Ignored
			Not used, since values will be projected.
		**params : dict
			Discarded since deprecated.

		Returns
		-------
		None
			Fits the model to the data.
		"""
		check_array(X)
		if y is not None:
			check_X_y(X, y)
			y = np.array(y)
		
		X = np.array(X)
		
		# Project time series onto vector space using only the normal data for
		# training since the One-class SVM requires to learn a boundary of
		# normal data.
		if y is not None:
			normal_data = np.argwhere(y == 0)
			normal_data = normal_data.reshape(normal_data.shape[0])
			normal_data = X[normal_data]
		else:
			normal_data = X

		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(normal_data)
		
		# Run vanilla OSVM on the vector space of the time series
		self._osvm()
		self.osvm.fit(X_new)
	
	def predict(self, X) -> np.ndarray:
		"""Predicts whether a sample is normal (0) or anomalous (1).
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data to be classified from the algorithm.

		Returns
		-------
		label : ndarray of shape (n_samples)
			The label associated to each point.
		"""
		check_array(X)
		X = np.array(X)
		
		# Project data onto the phase space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)
		
		# Predict labels for the phase space points (-1 is outlier and 1 inlier)
		# Therefore, I multiply by -1
		window_pred: np.ndarray = self.osvm.predict(X_new) * -1
		window_scores: np.ndarray = self.osvm.decision_function(X_new) * -1

		scorer = TimeSeriesAnomalyScorer(self.window,
										 self.stride,
										 self.scaling,
										 self.scoring)
		labeller = TimeSeriesAnomalyLabeller(self.window,
											 self.stride,
											 self.anomaly_threshold,
											 self.anomaly_contamination,
											 self.classification)
		scores = scorer.fit_transform(window_scores,
									  projector.num_windows_)
		labels, _ = labeller.fit_transform(window_pred,
										   projector.num_windows_,
										   scores=scores)
		
		return labels
	
	def anomaly_score(self, X):
		"""Returns the anomaly score of the points.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data to be classified from the algorithm.

		Returns
		-------
		scores : ndarray of shape (n_samples,)
			The scores for the points between 0 and 1. The higher, the more
			abnormal. If threshold is not None, points above threshold are
			labelled as anomalies. Otherwise, see how points are labelled.
		"""
		check_array(X)
		X = np.array(X)

		# Project data onto the phase space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Predict scores for the vector space points
		window_scores: np.ndarray = self.osvm.decision_function(X_new) * -1

		scorer = TimeSeriesAnomalyScorer(self.window,
										 self.stride,
										 self.scaling,
										 self.scoring)
		scores = scorer.fit_transform(window_scores,
									  projector.num_windows_)

		return scores
	
	def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : array-like of shape (n_samples, n_features), default=None
			If given, the training will be performed only on samples from the
			class with label 0 and each value must be either 0 or 1. If None,
			all the points in X are used as training.
		sample_weight : Ignored
			Not used, since values will be projected.

		Returns
		-------
		label : ndarray of shape (n_samples)
			The label associated to each point.
		"""
		self.fit(X, y, sample_weight)
		return self.predict(X)

	def _osvm(self):
		"""Initializes the wrapped OneClassSVM

		Returns
		-------
		None
		"""
		self.osvm = OneClassSVM(kernel=self.kernel,
								degree=self.degree,
								gamma=self.gamma,
								coef0=self.coef0,
								tol=self.tol,
								nu=self.nu,
								shrinking=self.shrinking,
								cache_size=self.cache_size,
								verbose=self.verbose,
								max_iter=self.max_iter)
