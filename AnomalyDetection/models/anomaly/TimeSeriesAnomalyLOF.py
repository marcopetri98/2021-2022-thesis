# Python imports
from typing import Union, Callable

# External imports
import numpy as np
import sklearn.neighbors as sk

# Project imports
from models.anomaly.TimeSeriesAnomalyWindowUnsupervised import TimeSeriesAnomalyWindowUnsupervised


class TimeSeriesAnomalyLOF(TimeSeriesAnomalyWindowUnsupervised):
	"""Concrete class representing the application of LOF approach to time series."""
	
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
				 anomaly_threshold: float = 0.5,
				 classification: str = "auto"):
		super().__init__(window,
						 stride,
						 anomaly_threshold,
						 classification)
		
		self.n_neighbors = n_neighbors
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.contamination = contamination
		self.novelty = novelty
		self.n_jobs = n_jobs
	
	def _fit_windowed_data(self, spatial_data: np.ndarray,
						   labels: np.ndarray = None,
						   num_points: int = 0,
						   *args,
						   **kwargs) -> None:
		"""Fit the model to the window and return the computed anomalies.
		
		Parameters
		----------
		spatial_data : ndarray
			A window of data on which to perform anomalies search.
		labels: ndarray
			Labels for the points of the dataset.
		num_points: int
			Number of points in the original dataset.
		
		Returns
		-------
		None
			The fitted model using the windowed spatial data.
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
		lof.fit(spatial_data)
		negative_outlier_factors = lof.negative_outlier_factor_
		
		anomalies_idx = np.argwhere(negative_outlier_factors < lof.offset_)
		self._windowed_anomalies = np.zeros(spatial_data.shape[0])
		self._windowed_anomalies[anomalies_idx] = 1
		self._windowed_scores = -negative_outlier_factors
