from typing import Union

import numpy as np
from numpy.random import RandomState
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array

from models.IParametric import IParametric
from models.time_series.anomaly.TimeSeriesAnomalyWindowWrapper import TimeSeriesAnomalyWindowWrapper


class TimeSeriesAnomalyIForest(TimeSeriesAnomalyWindowWrapper, IParametric):
	"""Isolation Forest adaptation to time series
        
    See Also
	--------
	For all the other parameters, see the scikit-learn implementation.
	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """

	def __init__(self, window: int = 5,
				 stride: int = 1,
				 scaling: str = "minmax",
				 scoring: str = "average",
				 classification: str = "voting",
				 threshold: float = None,
				 anomaly_portion: float = 0.01,
				 n_estimators: int = 100,
				 max_samples: Union[int, float, str] = 'auto',
				 contamination: Union[float, str] = 'auto',
				 max_features: Union[int, float] = 1.0,
				 bootstrap: bool = False,
				 n_jobs: int = None,
				 random_state: Union[int, RandomState] = None,
				 verbose: int = 0,
				 warm_start: bool = False):
		super().__init__(window=window,
						 stride=stride,
						 scaling=scaling,
						 scoring=scoring,
						 classification=classification,
						 threshold=threshold,
						 anomaly_portion=anomaly_portion)

		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.contamination = contamination
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose
		self.warm_start = warm_start
	
	def fit(self, X, y=None) -> None:
		check_array(X)
		X = np.array(X)
		
		x_new, windows_per_point = self.project_time_series(X)
		self.build_wrapped()
		self._wrapped_model.fit(x_new)
	
	def compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
		window_anomalies = self._wrapped_model.predict(vector_data) * -1
		return window_anomalies
	
	def compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
		window_scores = self._wrapped_model.decision_function(vector_data) * -1
		return window_scores

	def build_wrapped(self):
		self._wrapped_model = IsolationForest(n_estimators=self.n_estimators,
											  max_samples=self.max_samples,
											  contamination=self.contamination,
											  max_features=self.max_features,
											  bootstrap=self.bootstrap,
											  n_jobs=self.n_jobs,
											  random_state=self.random_state,
											  verbose=self.verbose,
											  warm_start=self.warm_start)