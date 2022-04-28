import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array, check_X_y

from input_validation.attribute_checks import check_not_default_attributes
from models.IParametric import IParametric
from models.time_series.anomaly.TimeSeriesAnomalyWindowWrapper import TimeSeriesAnomalyWindowWrapper


class TimeSeriesAnomalyOSVM(TimeSeriesAnomalyWindowWrapper, IParametric):
	"""OSVM adaptation to time series using scikit-learn.
        
    See Also
    --------
	For all the other parameters, see the scikit-learn implementation.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
	
	def __init__(self, window: int = 5,
				 stride: int = 1,
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 threshold: float = None,
				 anomaly_portion: float = 0.01,
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
		super().__init__(window=window,
						 stride=stride,
						 scaling=scaling,
						 scoring=scoring,
						 classification=classification,
						 threshold=threshold,
						 anomaly_portion=anomaly_portion)

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

	def fit(self, X, y=None, *args, **kwargs) -> None:
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
			
		x_new, windows_per_point = self._project_time_series(normal_data)
		self._build_wrapped()
		self._wrapped_model.fit(x_new)

	def anomaly_score(self, X, *args, **kwargs) -> np.ndarray:
		check_not_default_attributes(self, {"_wrapped_model": None})
		return super().anomaly_score(X)
	
	def classify(self, X, *args, **kwargs) -> np.ndarray:
		check_not_default_attributes(self, {"_wrapped_model": None})
		return super().classify(X)

	def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
		window_anomalies = self._wrapped_model.predict(vector_data) * -1
		return window_anomalies
	
	def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
		window_scores = self._wrapped_model.decision_function(vector_data) * -1
		return window_scores

	def _build_wrapped(self):
		self._wrapped_model = OneClassSVM(kernel=self.kernel,
										  degree=self.degree,
										  gamma=self.gamma,
										  coef0=self.coef0,
										  tol=self.tol,
										  nu=self.nu,
										  shrinking=self.shrinking,
										  cache_size=self.cache_size,
										  verbose=self.verbose,
										  max_iter=self.max_iter)
