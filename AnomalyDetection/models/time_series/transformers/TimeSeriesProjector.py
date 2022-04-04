# Python imports

# External imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

# Project imports


class TimeSeriesProjector(BaseEstimator, TransformerMixin):
	"""Transductive model projecting a time series into a vector space.
	
	Parameters
	----------
	window : int, default=200
		The dimension of the window used to project the time series onto a
		vector space.
	stride : int, default=1
		The dimension of the step to make in the time series to produce to next
		point of the vector space.
		
	Attributes
	----------
	num_windows_ : ndarray of shape (n_samples)
		The number of times a point has been used to produce a vector of the
		transformed space.
	n_features_in_ : int
		Number of features seen during fit.
	x_new_ : ndarray of shape (n_samples, window)
		The transformed data.
	"""
	
	def __init__(self, window: int,
				 stride: int):
		super().__init__()
		self.window = window
		self.stride = stride
	
	def fit_transform(self, time_series,
					  y=None,
					  **fit_params) -> np.ndarray:
		"""Compute the new space.

		Parameters
		----------
		time_series : array-like of shape (n_samples, n_features)
			The input data to be transformed.
		y : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		X_new : ndarray of shape (n_samples, window)
			The transformed data.
		"""
		# Input validation
		check_array(time_series)
		data = np.array(time_series)
		
		if self.window > data.shape[0]:
			raise ValueError("Window cannot be larger than data size.")
		elif data.shape[1] > 1:
			raise ValueError("Only univariate time series is currently "
							 "supported.")
		elif (data.shape[0] - self.window) % self.stride != 0:
			raise ValueError("Data.shape[0] - window must be a multiple of "
							 "stride to build the spatial data.")

		# Number of times a point is considered in a window
		self.n_features_in_ = data.shape[0]
		self.num_windows_ = np.zeros(data.shape[0])
		self.x_new_ = []
		
		# Transform univariate time series into spatial data
		for i in range(0, data.shape[0] - self.window + 1, self.stride):
			self.num_windows_[i:i + self.window] += 1
			current_data: np.ndarray = data[i:i + self.window]
			current_data = current_data.reshape(current_data.shape[0])
			self.x_new_.append(current_data.tolist())
			
		self.x_new_ = np.array(self.x_new_)
		
		return self.x_new_
