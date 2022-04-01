# Python imports
from typing import Union

# External imports
import numpy as np
from numpy.random import RandomState
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

# Project imports
from models.transformers.TimeSeriesAnomalyLabeller import \
	TimeSeriesAnomalyLabeller
from models.transformers.TimeSeriesAnomalyScorer import TimeSeriesAnomalyScorer
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyIForest(BaseEstimator, OutlierMixin):
	"""Isolation Forest adaptation to time series
	
	Parameters
    ----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
	
	anomaly_threshold: float, default=0.0
		The threshold used to compute if a point is an anomaly or not.

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
	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """

	def __init__(self, window: int = 200,
				 stride: int = 1,
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 anomaly_threshold: float = None,
				 anomaly_contamination: float = 0.01,
				 n_estimators: int = 100,
				 max_samples: Union[int, float, str] = 'auto',
				 contamination: Union[float, str] = 'auto',
				 max_features: Union[int, float] = 1.0,
				 bootstrap: bool = False,
				 n_jobs: int = None,
				 random_state: Union[int, RandomState] = None,
				 verbose: int = 0,
				 warm_start: bool = False):
		super().__init__()

		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.contamination = contamination
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose
		self.warm_start = warm_start

		self.iforest: IsolationForest = None
		self.window = window
		self.stride = stride
		self.classification = classification
		self.anomaly_threshold = anomaly_threshold
		self.scaling = scaling
		self.scoring = scoring
		self.anomaly_contamination = anomaly_contamination
		
	def _set_oob_score(self, X, y):
		raise NotImplementedError("OOB score not supported by iforest")
	
	def fit(self, X, y=None, sample_weight=None):
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.
		sample_weight : Ignored
			Not used, since values will be projected.

		Returns
		-------
		None
			Fits the model to the data.
		"""
		check_array(X)
		
		X = np.array(X)
		
		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)
		
		# Run vanilla
		self._iforest()
		self.iforest.fit(X_new)
		
	def predict(self, X):
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
		
		# Predict labels for the phase space points
		window_pred: np.ndarray = self.iforest.predict(X_new) * -1
		window_scores: np.ndarray = self.iforest.decision_function(X_new) * -1

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
		"""Return the anomaly scores for the given data.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data to be classified from the algorithm.

		Returns
		-------
		anomaly_scores : ndarray of shape (n_samples)
			The anomaly score associated to each point whose values are inside
			[0, 1]. The higher the value, the more probable it is an anomaly.
		"""
		check_array(X)
		X = np.array(X)

		# Project data onto the phase space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)

		# Predict scores for the vector space points
		window_scores: np.ndarray = self.iforest.decision_function(X_new) * -1

		scorer = TimeSeriesAnomalyScorer(self.window,
										 self.stride,
										 self.scaling,
										 self.scoring)
		scores = scorer.fit_transform(window_scores,
									  projector.num_windows_)
		
		return scores
		
	def fit_predict(self, X, y=None):
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		label : ndarray of shape (n_samples)
			The label associated to each point.
		"""
		self.fit(X, y)
		return self.predict(X)

	def _iforest(self):
		"""Initializes the wrapped IsolationForest.

		Returns
		-------
		None
		"""
		self.iforest = IsolationForest(n_estimators=self.n_estimators,
									   max_samples=self.max_samples,
									   contamination=self.contamination,
									   max_features=self.max_features,
									   bootstrap=self.bootstrap,
									   n_jobs=self.n_jobs,
									   random_state=self.random_state,
									   verbose=self.verbose,
									   warm_start=self.warm_start)
