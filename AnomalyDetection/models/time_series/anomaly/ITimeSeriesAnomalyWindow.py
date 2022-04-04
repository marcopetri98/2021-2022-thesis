import abc

import numpy as np

from models.time_series.anomaly.ITimeSeriesAnomaly import ITimeSeriesAnomaly


class ITimeSeriesAnomalyWindow(ITimeSeriesAnomaly):
	"""Interface for sliding window univariate time series anomaly detection.
    """
	
	@abc.abstractmethod
	def project_time_series(self, time_series: np.ndarray) -> np.ndarray:
		pass
	
	@abc.abstractmethod
	def compute_point_scores(self, window_scores,
							 windows_per_point) -> np.ndarray:
		pass
	
	@abc.abstractmethod
	def compute_point_labels(self, window_labels,
							 windows_per_point,
							 point_scores=None) -> np.ndarray:
		pass
	
	@abc.abstractmethod
	def compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
		pass
	
	@abc.abstractmethod
	def compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
		pass
