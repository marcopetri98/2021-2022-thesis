# Python imports
from typing import Tuple

# External imports
import numpy as np
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin

# Project imports
from input_validation.array_checks import check_x_y_smaller_1d


class TimeSeriesAnomalyLabeller(BaseEstimator, TransformerMixin):
	"""Transductive model computing labels from window labels.

	Parameters
	----------
	window : int
		The length of the windows over which the scoring must be performed.

	stride : int
		The stride used to project the time series on the window vector space.
		It is needed to correctly compute the point scores.

	threshold : float
		The threshold to be used to classify points. It must be between 0 and 1.
		When labelling_method="voting", it represents the percentage of voters
		who must agree in labelling. When labelling_method="points_score" it
		represents the threshold at which points are classified as anomalies.
		When it is None, if labelling_method="voting" than the threshold is 0.5
		and if labelling_method="points_score" the computed scores are used to
		fit a truncated normal distribution and threshold is defined as the
		quantile of 1-contamination.

	contamination : float, default=0.01
		It represents the percentage of anomaly points in the dataset. Moreover,
		it is used to compute the threshold when it is not specified to the
		labeller.

	labelling_method : str, default="voting"
		The method used to label points.
	"""
	ACCEPTED_LABELLING_METHODS = ["voting", "points_score"]

	def __init__(self, window: int,
				 stride: int,
				 threshold: float = None,
				 contamination: float = 0.01,
				 labelling_method: str = "voting"):
		if labelling_method not in self.ACCEPTED_LABELLING_METHODS:
			raise ValueError("Scoring method must be one of the following: " +
							 str(self.ACCEPTED_LABELLING_METHODS))
		elif window <= 0 or stride <= 0:
			raise ValueError("Stride and window must be positive.")
		elif threshold is not None and not 0 <= threshold <= 1:
			raise ValueError("Threshold must be None or 0 <= threshold <= 1")
		elif not 0 < contamination <= 0.5:
			raise ValueError("The contamination must be inside (0,0.5]")

		super().__init__()

		self.labelling_method = labelling_method
		self.window = window
		self.stride = stride
		self.threshold = threshold
		self.contamination = contamination

	def fit_transform(self, window_labels,
					  windows_per_point=None,
					  **fit_params) -> Tuple[np.ndarray, float]:
		"""Computes the scoring of the points for the time series.

		Parameters
		----------
		window_labels : array-like of shape (n_windows,)
			The labels of the windows for the time series to be used to compute
			the labels of the points.
			
		windows_per_point : array-like of shape (n_points,)
			The number of windows containing the point at that specific index.
			
		**fit_params
			Additional fit parameters. More details in Notes.

		Returns
		-------
		point_scores : ndarray
			The scores of the points.
		threshold : float
			The threshold above which points are considered as anomalies.

		Notes
		-----
		Here there is the list of all the parameters that can be passed using
		fit_params.

		* scores : ndarray of shape (n_points,)
			The scores of the points in range [0,1].
		"""
		check_x_y_smaller_1d(window_labels, windows_per_point)

		window_labels = np.array(window_labels)
		windows_per_point = np.array(windows_per_point)

		threshold = self.threshold
		labels = np.zeros(windows_per_point.shape[0])
		match self.labelling_method:
			case "voting":
				# Anomalies are computed by voting of window anomalies
				for i in range(window_labels.shape[0]):
					if window_labels[i] == 1:
						idx = i * self.stride
						labels[idx:idx + self.window] += 1
				labels = labels / windows_per_point

				if threshold is None:
					threshold = 0.5

				true_anomalies = np.argwhere(labels > threshold)
				labels = np.zeros(labels.shape)
				labels[true_anomalies] = 1

			case "points_score":
				# Anomalies are computed over the point scores
				scores = fit_params["scores"]

				# Computes the threshold using the 99 percentile
				if threshold is None:
					mean = np.mean(scores)
					std = np.std(scores)
					a, b = (0 - mean) / std, (1 - mean) / std
					threshold = truncnorm.ppf(1 - self.contamination,
											  a,
											  b,
											  loc=mean,
											  scale=std)

				labels[np.argwhere(scores > threshold)] = 1

		return labels, threshold
