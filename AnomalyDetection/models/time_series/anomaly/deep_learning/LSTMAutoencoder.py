# Python imports
from typing import Tuple

# External imports
import tensorflow as tf

# Project imports
from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyAutoencoder import TimeSeriesAnomalyAutoencoder


class LSTMAutoencoder(TimeSeriesAnomalyAutoencoder):
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
		return self._learning_create_model(input_shape)
	
	def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		input_layer = tf.keras.layers.Input(input_shape,
											name="input")
		
		encoder = tf.keras.layers.LSTM(32,
									   return_sequences=True,
									   name="encoder_lstm_1")(input_layer)
		encoder = tf.keras.layers.LSTM(16,
									   return_sequences=True,
									   name="encoder_lstm_2")(encoder)
		
		middle = tf.keras.layers.LSTM(8,
									  name="encoder_lstm_3")(encoder)
		middle = tf.keras.layers.RepeatVector(self.window)(middle)
		
		decoder = tf.keras.layers.LSTM(16,
									   return_sequences=True,
									   name="decoder_lstm_1")(middle)
		decoder = tf.keras.layers.LSTM(32,
									   return_sequences=True,
									   name="decoder_lstm_2")(decoder)
		
		dense = tf.keras.layers.Dense(input_shape[1],
									  activation="linear",
									  name="output")
		output_layer = tf.keras.layers.TimeDistributed(dense)(decoder)
		
		model = tf.keras.Model(inputs=input_layer,
							   outputs=output_layer,
							   name="lstm_autoencoder")
		
		model.compile(loss="mse",
					  optimizer=tf.keras.optimizers.Adam(),
					  metrics=[tf.keras.metrics.MeanSquaredError()])
		
		return model