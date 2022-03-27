# Python imports
from typing import Union

# External imports
import numpy as np
from numpy.random import RandomState
from sklearn.base import OutlierMixin
from sklearn.ensemble import IsolationForest

# Project imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyIForest(IsolationForest, OutlierMixin):
	"""Isolation Forest adaptation to time series
	
	Parameters
    ----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
	
	anomaly_threshold: float, default=0.0
		The threshold used to compute if a point is an anomaly or not.
	
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
        
    See Also
	--------
	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
	
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 anomaly_threshold: float = 0,
				 n_estimators: int = 100,
				 max_samples: Union[int, float, str] = 'auto',
				 contamination: Union[float, str] = 'auto',
				 max_features: Union[int, float] = 1.0,
				 bootstrap: bool = False,
				 n_jobs: int = None,
				 random_state: Union[int, RandomState] = None,
				 verbose: int = 0,
				 warm_start: bool = False):
		super().__init__(n_estimators=n_estimators,
						 max_samples=max_samples,
						 contamination=contamination,
						 max_features=max_features,
						 bootstrap=bootstrap,
						 n_jobs=n_jobs,
						 random_state=random_state,
						 verbose=verbose,
						 warm_start=warm_start)
		
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold
		
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
		super().fit(X_new)
		
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
		window_pred: np.ndarray = super().predict(X_new) * -1
		
		# TODO: partially identical to DBSCAN code, find a way to avoid code duplication
		predictions = np.zeros(X.shape[0])
		
		# Anomalies are computed by voting of window anomalies
		for i in range(window_pred.shape[0]):
			if window_pred[i] == 1:
				idx = i * self.stride
				predictions[idx:idx + self.window] += 1
		predictions = predictions / projector.num_windows_
		
		true_anomalies = np.argwhere(predictions > self.anomaly_threshold)
		predictions = np.zeros(predictions.shape)
		predictions[true_anomalies] = 1
		
		return predictions
	
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
		
		# Predict labels for the phase space points
		window_scores: np.ndarray = super().decision_function(X_new) * -1
		window_scores = window_scores.reshape((window_scores.shape[0], 1))
		window_scores = MinMaxScaler().fit_transform(window_scores)
		window_scores = window_scores.reshape(window_scores.shape[0])
		
		# TODO: partially identical to DBSCAN code, find a way to avoid code duplication
		point_scores = np.zeros(X.shape[0])
		
		# Compute score of each point
		for i in range(window_scores.shape[0]):
			idx = i * self.stride
			point_scores[idx:idx + self.window] += window_scores[i]
		point_scores = point_scores / projector.num_windows_
		
		# Min-max normalization
		point_scores = point_scores.reshape((point_scores.shape[0], 1))
		point_scores = MinMaxScaler().fit_transform(point_scores)
		point_scores = point_scores.reshape(point_scores.shape[0])
		
		return point_scores
		
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
