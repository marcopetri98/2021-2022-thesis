# Python imports
import abc
from typing import Tuple

# External imports
import numpy as np
import tensorflow as tf

# Project imports
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyWindowDL import TimeSeriesAnomalyWindowDL


class TimeSeriesAnomalyAutoregressive(TimeSeriesAnomalyWindowDL):
	"""TimeSeriesAnomalyAutoregressive"""
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 forecast: int = 1,
				 batch_size: int = 32,
				 max_epochs: int = 50,
				 predict_validation: float = 0.2,
				 batch_divide_training: bool = False,
				 folder_save_path: str = "nn_models/",
				 filename: str = "lstm"):
		super().__init__(window,
						 stride,
						 forecast,
						 batch_size,
						 max_epochs,
						 predict_validation,
						 batch_divide_training,
						 folder_save_path,
						 filename)
	
	def _build_x_y_sequences(self, X) -> Tuple[np.ndarray, np.ndarray]:
		"""Build the neural network inputs to perform regression.

		Parameters
		----------
		X : np.ndarray
			The training input sequence.

		Returns
		-------
		x_train : np.ndarray
			Sequences of training samples to use as training.
		y_train : np.ndarray
			Targets of each training sample to use.
		"""
		samples = []
		targets = []
		
		for i in range(0, X.shape[0] - self.window - self.forecast,
					   self.stride):
			samples.append(X[i:i + self.window])
			targets.append(X[i + self.window:i + self.window + self.forecast])
		
		return np.array(samples), np.array(targets)
	
	def _predict_future(self, X: np.ndarray, points: int) -> np.ndarray:
		"""Starting from the window X, it predicts the next N points.

		It predicts points in an autoregressive way using the class forecast
		dimension.

		Parameters
		----------
		X : ndarray of shape (window, n_features)
			The window from which we have to predict the next samples.

		points : int
			The number of points we need to predict.

		Returns
		-------
		predicted_values : ndarray of shape (points, n_features)
			The predicted values for the next points.
		"""
		check_array(X)
		X = np.array(X)
		
		predictions = np.array([])
		for _ in range(0, points, self.forecast):
			# Make prediction assuming shape as (forecast, features)
			input_ = X.reshape((1, X.shape[0], X.shape[1]))
			prediction = self.model_.predict(input_)
			not_batch_output = prediction.reshape((self.forecast, X.shape[1]))
			
			if len(predictions) == 0:
				predictions = not_batch_output
			else:
				predictions = np.concatenate((predictions, not_batch_output))
			
			# Autoregressive step
			# TODO: ERRORE
			X = np.concatenate((X[self.forecast:], not_batch_output), axis=0)
		
		if predictions.shape[0] > points:
			to_discard = predictions.shape[0] - points
			predictions = predictions[:-to_discard]
		
		return predictions
	
	def predict_time_series(self, Xp, X) -> np.ndarray:
		"""Predict the future values of the time series.

		Parameters
		----------
		Xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		X : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		labels : ndarray
			The values of the steps predicted from the time series.
		"""
		check_is_fitted(self, ["threshold_", "model_"])
		check_array(Xp)
		check_array(X)
		Xp = np.array(Xp)
		X = np.array(X)
		
		if Xp.shape[0] < self.window:
			raise ValueError("You must provide at lest window points to predict")
		
		return self._predict_future(X[-self.window:], X.shape[0])
	
	@abc.abstractmethod
	def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		pass
	
	@abc.abstractmethod
	def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		pass
