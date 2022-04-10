from abc import ABC
from typing import Tuple

import numpy as np
from keras.callbacks import History
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyWindowDL import TimeSeriesAnomalyWindowDL
from print_utils.printing import print_warning


class TimeSeriesAnomalyAutoencoder(TimeSeriesAnomalyWindowDL, ABC):
	"""A framework model for autoencoders for anomaly detection
	
	Parameters
	----------
	extend_not_multiple : bool, default=True
		States whether the training and validation sets must be resized in case
		they are not a multiple of window. If true, exceeding elements will be
		discarded to train and validate on multiple of window size.
	"""
	
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
		check_X_y(x, y)
		x = np.array(x)
		y = np.array(y)
		
		if self.extend_not_multiple:
			print_warning("Data will be modified to match the window size, note"
						  " that the input must be a multiple of the window "
						  "size for autoencoders. This is just a comfort "
						  "utility. You can ignore this warning if no other "
						  "warnings regarding the training set pop up.")
			
			for i in range(len(training_idx)):
				slice_ = training_idx[i]
				if (slice_.stop - slice_.start) % self.window != 0:
					remainder = (slice_.stop - slice_.start) % self.window
					training_idx[i] = slice(slice_.start, slice_.stop - remainder)
					
					print_warning("On ", i+1, "th training slice, ", remainder,
								  " points have been lost")
					
			for i in range(len(validation_idx)):
				slice_ = validation_idx[i]
				if (slice_.stop - slice_.start) % self.window != 0:
					remainder = (slice_.stop - slice_.start) % self.window
					validation_idx[i] = slice(slice_.start, slice_.stop - remainder)
					
					print_warning("On ", i+1, "th validation slice, ",
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
	
	def _predict_future(self, xp: np.ndarray, x: np.ndarray) -> np.ndarray:
		check_is_fitted(self, ["model_"])
		check_array(x)
		x = np.array(x)
		
		if x.shape[0] % self.window != 0:
			raise ValueError("The autoencoder can predict only a number of "
							 "points such that points % self.window == 0")
			
		# I build the array of inputs for the model
		inputs = []
		for i in range(0, x.shape[0], self.window):
			next_input = x[i:i + self.window]
			inputs.append(next_input.copy())
		inputs = np.array(inputs)
		
		predictions = self.model_.predict(inputs, batch_size=1)
		# The number of predicted points is batches * input.shape[0]
		num_points = predictions.shape[0] * predictions.shape[1]
		predictions = predictions.reshape((num_points, 1))
		
		return predictions
	
	def predict_time_series(self, xp, x) -> np.ndarray:
		check_is_fitted(self, ["model_"])
		check_array(x)
		x = np.array(x)
		
		if x.shape[0] < self.window:
			raise ValueError("You must provide at lest window points to predict")
		elif x.shape[0] % self.window != 0:
			raise ValueError("An autoencoder must receive as input a multiple "
							 "of window to be able to predict. Namely, the input"
							 "must be such that X.shape[0] % self.window == 0")
		
		return self._predict_future(xp, x)
