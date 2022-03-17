# Python imports
from typing import Union, Callable, Tuple

# External imports
import numpy as np
import sklearn.neighbors as sk

# Project imports
from models.anomaly.AnomalyWindowUnsupervised import AnomalyWindowUnsupervised


class TimeSeriesLOF(AnomalyWindowUnsupervised):
	"""Concrete class representing the application of LOF approach to time series."""
	ERROR_KEY = AnomalyWindowUnsupervised.ERROR_KEY.copy() + ["lof"]
	
	def __init__(self, n_neighbors: int = 20,
				 algorithm: str = 'auto',
				 leaf_size: int = 30,
				 metric: Union[str, Callable[[list, list], float]] = 'minkowski',
				 p: int = 2,
				 metric_params: dict = None,
				 contamination: Union[str, float] = 'auto',
				 novelty: bool = False,
				 n_jobs: int = None,
				 window: int = 200,
				 stride: int = 200,
				 anomaly_threshold: float = 1.5):
		super().__init__(window, stride, anomaly_threshold)
		
		self.n_neighbors = n_neighbors
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.contamination = contamination
		self.novelty = novelty
		self.n_jobs = n_jobs
	
	def fit(self, train: np.ndarray,
			*args,
			**kwargs) -> None:
		"""Fit the LOF model to the time series data using scikit-learn.

		Parameters
		----------
		train : ndarray of shape (n_samples, n_features)
			The time series data without containing the index, timestmap or not.

		Returns
		-------
		None
		"""
		super().fit(train, None, args, kwargs)
	
	def _compute_anomalies(self, num_evaluations: np.ndarray,
						   *args,
						   **kwargs) -> None:
		"""Compute which are the samples categorized as anomaly."""
		anomalies = np.argwhere(self.anomaly_scores > self.anomaly_threshold)
		self.anomalies = np.zeros(self.anomalies.shape)
		self.anomalies[anomalies] = 1
	
	def _fit_window(self, window_data: np.ndarray,
					*args,
					**kwargs) -> Tuple[np.ndarray]:
		"""Fit the dbscan to the window and return the computed anomalies.

		Parameters
		----------
		window_data : ndarray
			A window of data on which to perform anomalies search.

		Returns
		-------
		anomaly_scores : ndarray of shape window_data.shape[0]
			The anomaly scores for the points of the window dataset.
		"""
		lof = sk.LocalOutlierFactor(self.n_neighbors,
									algorithm=self.algorithm,
									leaf_size=self.leaf_size,
									metric=self.metric,
									p=self.p,
									metric_params=self.metric_params,
									contamination=self.contamination,
									novelty=self.novelty,
									n_jobs=self.n_jobs)
		lof.fit(window_data)
		
		if self.anomaly_threshold == 1.5:
			self.anomaly_threshold = - lof.offset_
		
		return - lof.negative_outlier_factor_
