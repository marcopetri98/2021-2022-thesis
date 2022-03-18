# Python imports
from typing import Union, Callable

# External imports
import numpy as np
import sklearn.svm as sk

# Project imports
from models.Predictor import Predictor
from models.anomaly.AnomalyLearner import AnomalyLearner, project_time_series


class TimeSeriesAnomalyOSVM(AnomalyLearner, Predictor):
	"""Concrete class representing the application of LOF approach to time series."""
	
	def __init__(self, kernel: str = "rbf",
				 degree: int = 3,
				 gamma: str = "scale",
				 coef0: float = 0.0,
				 tol: float = 1e-3,
				 nu: float = 0.5,
				 shrinking: bool = True,
				 cache_size: float = 200,
				 verbose: bool = False,
				 max_iter: int = -1,
				 window: int = 200,
				 stride: int = 200,
				 anomaly_threshold: float = 0):
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
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold
		self.osvm = None
	
	def fit(self, train: np.ndarray,
			labels: np.ndarray = None,
			*args,
			**kwargs) -> None:
		#normal_data = np.argwhere(labels == 0)
		training = train#[normal_data]
		#training = training.reshape((training.shape[0], training.shape[1]))
		spatial_time_series, num_evaluations = project_time_series(self.window,
																   self.stride,
																   training)
		self.osvm = sk.OneClassSVM(kernel=self.kernel,
								   degree=self.degree,
								   gamma=self.gamma,
								   coef0=self.coef0,
								   tol=self.tol,
								   nu=self.nu,
								   shrinking=self.shrinking,
								   cache_size=self.cache_size,
								   verbose=self.verbose,
								   max_iter=self.max_iter)
		self.osvm.fit(spatial_time_series)
	
	def predict(self, test: np.ndarray,
			*args,
			**kwargs) -> np.ndarray:
		spatial_time_series, num_evaluations = project_time_series(self.window,
																   self.stride,
																   test)
		
		outlier_score = - self.osvm.decision_function(spatial_time_series)
		
		# Compute the anomaly score of the original points as average of windows
		self.anomaly_scores = np.zeros(test.shape[0])
		for i in range(outlier_score.shape[0]):
			idx = i * self.stride
			self.anomaly_scores[idx:idx + self.window] += outlier_score[i]
		self.anomaly_scores = self.anomaly_scores / num_evaluations

		anomalies = np.argwhere(self.anomaly_scores > self.anomaly_threshold)
		self.anomalies = np.zeros(test.shape[0])
		self.anomalies[anomalies] = 1
		
		return self.anomalies
