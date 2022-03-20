# Python imports

# External imports
import numpy as np
from sklearn.base import OutlierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array, check_X_y

# Project imports
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyOSVM(OneClassSVM, OutlierMixin):
	"""OSVM adaptation to time series using scikit-learn.

    Parameters
    ----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
	
	classification: {"voting", "points_score"}, default="voting"
		It defines the way in which a point is declared as anomaly. With voting,
		a point is an anomaly if at least anomaly_threshold percentage of
		windows containing the point agree in saying it is an anomaly. With
		points_score, the points are considered anomalies if they're score is
		above anomaly_threshold.
	
	anomaly_threshold: float, default=0.0
		The threshold used to compute if a point is an anomaly or not.
	
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
        
    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 anomaly_threshold: float = 0,
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
		super().__init__(kernel=kernel,
						 degree=degree,
						 gamma=gamma,
						 coef0=coef0,
						 tol=tol,
						 nu=nu,
						 shrinking=shrinking,
						 cache_size=cache_size,
						 verbose=verbose,
						 max_iter=max_iter)
		
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold
	
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
			check_X_y(X,y)
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
		super().fit(X_new)
	
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
		
		# Predict labels for the phase space points
		window_pred: np.ndarray = super().predict(X_new)

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
	
	def score_samples(self, X):
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
		window_scores: np.ndarray = super().decision_function(X_new) * (-1)
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
