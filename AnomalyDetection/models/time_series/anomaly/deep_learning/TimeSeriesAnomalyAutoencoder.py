import abc
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.callbacks import History
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyWindowDL import TimeSeriesAnomalyWindowDL


class TimeSeriesAnomalyAutoencoder(TimeSeriesAnomalyWindowDL):
	"""TimeSeriesAnomalyAutoencoder"""
	
	def __init__(self, window: int = 200,
				 forecast: int = 1,
				 batch_size: int = 32,
				 max_epochs: int = 50,
				 predict_validation: float = 0.2,
				 batch_divide_training: bool = False,
				 folder_save_path: str = "nn_models/",
				 filename: str = "lstm",
				 extend_not_multiple: bool = True):
		super().__init__(window,
						 window,
						 forecast,
						 batch_size,
						 max_epochs,
						 predict_validation,
						 batch_divide_training,
						 folder_save_path,
						 filename)
		
		self.extend_not_multiple = extend_not_multiple
	
	def fit(self, x, training_idx, validation_idx, y) -> list[History]:
		"""Train the predictor and the threshold using a simple Perceptron.

		Parameters
		----------
		x : array-like of ndarray of shape (n_samples, n_features)
			Data on which the predictor is trained to be able to learn a model
			capable of providing good prediction performances and on which it
			is validated to learn the threshold to evaluate if a point is an
			anomaly or not.

		training_idx : list of slice objects
			A list of the slice to apply on ``X`` to retrieve the training
			sequences.

		validation_idx : list of slice objects
			A list of the slice to apply on ``X`` to retrieve the validation set
			to learn the threshold.

		y : array-like of shape (n_samples, n_features)
			Data labels with shape consistent with X used to learn
			the decision boundary for the anomalies.

		Returns
		-------
		histories: list of History
			The list of the training history for the predictor.
		"""
		check_X_y(x, y)
		x = np.array(x)
		y = np.array(y)
		
		if self.extend_not_multiple:
			print("WARNING: Data will be reduced because of extension option, "
				  "however, the training and validation data will be modified.")
			
			for i in range(len(training_idx)):
				slice_ = training_idx[i]
				if (slice_.stop - slice_.start) % self.window != 0:
					remainder = (slice_.stop - slice_.start) % self.window
					training_idx[i] = slice(slice_.start, slice_.stop - remainder)
					
					print("WARNING: On ", i+1, "th training slice, ", remainder,
						  " points have been lost")
					
			for i in range(len(validation_idx)):
				slice_ = validation_idx[i]
				if (slice_.stop - slice_.start) % self.window != 0:
					remainder = (slice_.stop - slice_.start) % self.window
					validation_idx[i] = slice(slice_.start, slice_.stop - remainder)
					
					print("WARNING: On ", i+1, "th validation slice, ",
						  remainder, " points have been lost")
					
		history = super().fit(x, training_idx, validation_idx, y)
		return history
	
	def _build_x_y_sequences(self, x) -> Tuple[np.ndarray, np.ndarray]:
		samples = []
		targets = []
		
		for i in range(0, x.shape[0] - self.window, self.stride):
			samples.append(x[i:i + self.window])
			targets.append(x[i:i + self.window])
		
		return np.array(samples), np.array(targets)
	
	def _predict_future(self, xp: np.ndarray, x: int) -> np.ndarray:
		"""Starting from the window X, it predicts the next N points.

		It predicts points window by window being an autoencoder.

		Parameters
		----------
		xp : ndarray of shape (window, n_features)
			The data to be predicted from the autoencoder.

		x : int
			The number of points we need to predict.

		Returns
		-------
		predicted_values : ndarray of shape (points, n_features)
			The predicted values for the next points.
		"""
		check_is_fitted(self, ["threshold_"])
		check_array(xp)
		xp = np.array(xp)
		
		if x % self.window != 0:
			raise ValueError("The autoencoder can predict only a number of "
							 "points such that points % self.window == 0")
		
		predictions = np.array([])
		for idx in range(0, x, self.stride):
			# Make prediction window by window
			input_ = xp[idx:idx + self.window]
			input_ = input_.reshape((1, xp.shape[0], xp.shape[1]))
			prediction = self.model_.predict(input_, )
			not_batch_output = prediction.reshape((self.window, xp.shape[1]))
			
			if len(predictions) == 0:
				predictions = not_batch_output
			else:
				predictions = np.concatenate((predictions, not_batch_output))
		
		if predictions.shape[0] > x:
			to_discard = predictions.shape[0] - x
			predictions = predictions[:-to_discard]
		
		return predictions
	
	def predict_time_series(self, xp, x) -> np.ndarray:
		"""Predict the future values of the time series.

		Parameters
		----------
		xp : Ignored
			Present because of API consistency.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		labels : ndarray
			The values of the steps predicted from the time series.
		"""
		check_is_fitted(self, ["model_"])
		check_array(x)
		x = np.array(x)
		
		if x.shape[0] < self.window:
			raise ValueError("You must provide at lest window points to predict")
		elif x.shape[0] % self.window != 0:
			raise ValueError("An autoencoder must receive as input a multiple "
							 "of window to be able to predict. Namely, the input"
							 "must be such that X.shape[0] % self.window == 0")
		
		return self._predict_future(x, x.shape[0])
	
	@abc.abstractmethod
	def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		pass
	
	@abc.abstractmethod
	def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		pass
