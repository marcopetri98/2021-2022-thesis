from abc import ABC

import numpy as np
from sklearn.utils import check_array

from models.time_series.anomaly.deep_learning.TimeSeriesAnomalySequential import TimeSeriesAnomalySequential


class TimeSeriesAnomalyAutoregressive(TimeSeriesAnomalySequential, ABC):
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
	
	def _predict_future(self, xp: np.ndarray, x: int) -> np.ndarray:
		check_is_fitted(self, ["model_"])
		check_array(xp)
		xp = np.array(xp)
		
		predictions = np.array([])
		for _ in range(0, xp, self.forecast):
			# Make prediction assuming shape as (forecast, features)
			input_ = xp.reshape((1, xp.shape[0], xp.shape[1]))
			prediction = self.model_.predict(input_, )
			not_batch_output = prediction.reshape((self.forecast, xp.shape[1]))
			
			if len(predictions) == 0:
				predictions = not_batch_output
			else:
				predictions = np.concatenate((predictions, not_batch_output))
			
			# Autoregressive step
			xp = np.concatenate((xp[self.forecast:], not_batch_output), axis=0)
		
		if predictions.shape[0] > xp:
			to_discard = predictions.shape[0] - xp
			predictions = predictions[:-to_discard]
		
		return predictions
