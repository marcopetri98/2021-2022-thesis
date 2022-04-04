# Python imports

# External imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

# Project imports
from input_validation.array_checks import check_x_y_smaller_1d


class TimeSeriesAnomalyScorer(BaseEstimator, TransformerMixin):
	"""Transductive model computing anomaly score from window scores.

	Parameters
	----------
	window : int
		The length of the windows over which the scoring must be performed.
		
	stride : int
		The stride used to project the time series on the window vector space.
		It is needed to correctly compute the point scores.
	
	scaling_method : str
		The method used to scale anomaly scores.
	
	scoring_method : str
		The method used to compute the anomaly scores of the points.
	"""
	ACCEPTED_SCORING_METHODS = ["average"]
	ACCEPTED_SCALING_METHODS = ["none", "minmax"]

	def __init__(self, window: int,
				 stride: int,
				 scaling_method: str = "minmax",
				 scoring_method: str = "average"):
		if scoring_method not in self.ACCEPTED_SCORING_METHODS:
			raise ValueError("Scoring method must be one of the following: " +
							 str(self.ACCEPTED_SCORING_METHODS))
		elif scaling_method not in self.ACCEPTED_SCALING_METHODS:
			raise ValueError("Scoring method must be one of the following: " +
							 str(self.ACCEPTED_SCALING_METHODS))
		elif window <= 0 or stride <= 0:
			raise ValueError("Stride and window must be positive.")

		super().__init__()

		self.scoring_method = scoring_method
		self.scaling_method = scaling_method
		self.window = window
		self.stride = stride

	def fit_transform(self, window_scores,
					  windows_per_point=None,
					  **fit_params) -> np.ndarray:
		"""Computes the scoring of the points for the time series.

		Parameters
		----------
		window_scores : array-like of shape (n_windows,)
			The scores of the windows for the time series to be used to compute
			the scores of the points.
		windows_per_point : array-like of shape (n_points,)
			The number of windows containing the point at that specific index.
		fit_params : Ignored
			Not used, present for API consistency by convention.

		Returns
		-------
		point_scores : ndarray
			The scores of the points.
		"""
		check_x_y_smaller_1d(window_scores, windows_per_point)

		window_scores = np.array(window_scores)
		windows_per_point = np.array(windows_per_point)

		# Compute score of each point
		scores = np.zeros(windows_per_point.shape[0])
		for i in range(window_scores.shape[0]):
			idx = i * self.stride
			scores[idx:idx + self.window] += window_scores[i]

		match self.scoring_method:
			case "average":
				scores = scores / windows_per_point

		match self.scaling_method:
			case "minmax":
				# Min-max normalization
				scores = scores.reshape((scores.shape[0], 1))
				scores = MinMaxScaler().fit_transform(scores)
				scores = scores.reshape(scores.shape[0])

		return scores
