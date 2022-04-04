# Python imports
from typing import Tuple

# External imports
import tensorflow as tf

# Project imports
from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyAutoencoder import TimeSeriesAnomalyAutoencoder


class TimeSeriesAnomalyLSTMAutoencoder(TimeSeriesAnomalyAutoencoder):
	"""LSTM model to identify anomalies in time series."""
	
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
						 forecast,
						 batch_size,
						 max_epochs,
						 predict_validation,
						 batch_divide_training,
						 folder_save_path,
						 filename,
						 extend_not_multiple)
	
	def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		"""Creates the LSTM model to perform the predictions.

		This function should be used in case the prediction is to be performed
		with batch=1 while the training is done with fixed batch and a batch
		greater than 1 (the prediction batch). If the batch is not fixed in
		training, this function should call ``_learning_create_model``.

		Returns
		-------
		model : tf.keras.Model
			The model for the prediction.
		"""
		return self._learning_create_model(input_shape)
	
	def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		"""Creates the LSTM model to perform the training.

		Returns
		-------
		model : tf.keras.Model
			The model for the prediction.
		"""
		
		input_layer = tf.keras.layers.Input(input_shape,
											name="input")
		
		encoder = tf.keras.layers.LSTM(32,
									   return_sequences=True,
									   activation="relu",
									   name="encoder_lstm_1")(input_layer)
		
		encoder = tf.keras.layers.LSTM(16,
									   activation="relu",
									   name="encoder_lstm_2")(encoder)
		
		middle = tf.keras.layers.RepeatVector(self.window)(encoder)
		
		decoder = tf.keras.layers.LSTM(16,
									   return_sequences=True,
									   activation="relu",
									   name="decoder_lstm_1")(middle)
		
		decoder = tf.keras.layers.LSTM(32,
									   activation="relu",
									   name="decoder_lstm_2")(decoder)
		
		output = tf.keras.layers.TimeDistributed()
		
		output_layer = tf.keras.layers.Dense(self.window * input_shape[1],
											 name="output",
											 activation="linear")(decoder)
		
		output_layer = tf.keras.layers.Reshape((self.window, input_shape[1]),
											   name="reshape")(output_layer)
		
		model = tf.keras.Model(inputs=input_layer,
							   outputs=output_layer,
							   name="lstm_autoencoder")
		
		model.compile(loss="mse",
					  optimizer=tf.keras.optimizers.Adam(),
					  metrics=[tf.keras.metrics.MeanSquaredError()])
		
		return model