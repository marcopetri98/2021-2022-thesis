# Python imports

# External imports
import numpy as np
from sklearn.base import OutlierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array, check_X_y

# Project imports
from sklearn.utils.validation import check_is_fitted

from models.time_series.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyOSVMPhase(OneClassSVM, OutlierMixin):
	"""OSVM adaptation to time series using scikit-learn.

    Parameters
    ----------
	window : int
		The length of the window to consider performing anomaly detection.

	stride : int
		The offset at which the window is moved when computing the anomalies.

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
	
	def __init__(self, windows: list[int] = None,
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
		
		if windows is None:
			windows = [200]
		
		self.windows = windows
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
		
		self.predictors_ = []
		for embedding in self.windows:
			projector = TimeSeriesProjector(embedding, self.stride)
			X_new = projector.fit_transform(normal_data)
			
			# Run vanilla OSVM on the vector space of the time series
			osvm = OneClassSVM(kernel=self.kernel,
							   degree=self.degree,
							   gamma=self.gamma,
							   coef0=self.coef0,
							   tol=self.tol,
							   nu=self.nu,
							   shrinking=self.shrinking,
							   cache_size=self.cache_size,
							   verbose=self.verbose,
							   max_iter=self.max_iter)
			osvm.fit(X_new)
			self.predictors_.append(osvm)
	
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
		check_is_fitted(self, ["predictors_"])
		check_array(X)
		X = np.array(X)
		
		predictions = []
		for i in range(len(self.windows)):
			# Project data onto the phase space
			projector = TimeSeriesProjector(self.windows[i], self.stride)
			X_new = projector.fit_transform(X)
			
			# Predict labels for the phase space points
			window_pred: np.ndarray = self.predictors_[i].predict(X_new)
			
			predictions.append(np.zeros(X.shape[0]))
			
			# Anomalies are computed by voting of window anomalies
			for j in range(window_pred.shape[0]):
				if window_pred[j] == 1:
					idx = j * self.stride
					predictions[i][idx:idx + self.windows[i]] = 1
		
		anomaly_votes = np.zeros(X.shape[0])
		for i in range(len(self.windows)):
			averaged_votes = predictions[i] / len(self.windows)
			anomaly_votes += averaged_votes
		
		true_anomalies = np.argwhere(anomaly_votes > self.anomaly_threshold)
		anomaly_votes = np.zeros(anomaly_votes.shape)
		anomaly_votes[true_anomalies] = 1
		
		return anomaly_votes
	
	def decision_function(self, X):
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
		check_is_fitted(self, ["predictors_"])
		check_array(X)
		X = np.array(X)
		
		scores = []
		for i in range(len(self.windows)):
			# Project data onto the phase space
			projector = TimeSeriesProjector(self.windows[i], self.stride)
			X_new = projector.fit_transform(X)
			
			# Predict labels for the phase space points
			window_scores: np.ndarray = self.predictors_[i].decision_function(
				X_new) * (-1)
			window_scores = window_scores.reshape((window_scores.shape[0], 1))
			window_scores = MinMaxScaler().fit_transform(window_scores)
			window_scores = window_scores.reshape(window_scores.shape[0])
			
			scores.append(np.zeros(X.shape[0]))
			# Anomalies are computed by voting of window anomalies
			for j in range(window_scores.shape[0]):
				idx = j * self.stride
				scores[i][idx:idx + self.windows[i]] += window_scores[i]
			scores[i] = scores[i] / projector.num_windows_
		
		anomaly_scores = np.zeros(X.shape[0])
		for i in range(len(self.windows)):
			averaged_votes = scores[i] / len(self.windows)
			anomaly_scores += averaged_votes
		
		# Min-max normalization
		anomaly_scores = anomaly_scores.reshape((anomaly_scores.shape[0], 1))
		anomaly_scores = MinMaxScaler().fit_transform(anomaly_scores)
		anomaly_scores = anomaly_scores.reshape(anomaly_scores.shape[0])
		
		return anomaly_scores
	
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
