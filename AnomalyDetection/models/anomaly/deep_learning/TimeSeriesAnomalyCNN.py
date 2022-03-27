# Python imports
from typing import Tuple

# External imports
import tensorflow as tf

# Project imports
from models.anomaly.deep_learning.TimeSeriesAnomalyAutoregressive import TimeSeriesAnomalyAutoregressive


class TimeSeriesAnomalyCNN(TimeSeriesAnomalyAutoregressive):
	"""TimeSeriesAnomalyCNN."""
	
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
			The model for the learning.
		"""
		
		input_layer = tf.keras.layers.Input(input_shape,
											name="input")
		
		cnn_model = tf.keras.layers.Conv1D(8, 2,
										   activation="relu",
										   name="conv1d_1")(input_layer)
		cnn_model = tf.keras.layers.MaxPool1D(2,
											  name="maxpool1d_1")(cnn_model)
		cnn_model = tf.keras.layers.Conv1D(16, 2,
										   activation="relu",
										   name="conv1d_2")(cnn_model)
		cnn_model = tf.keras.layers.MaxPool1D(2,
											  name="maxpool1d_2")(cnn_model)
		cnn_model = tf.keras.layers.Conv1D(32, 2,
										   activation="relu",
										   name="conv1d_3")(cnn_model)
		cnn_model = tf.keras.layers.MaxPool1D(2,
											  name="maxpool1d_3")(cnn_model)
		cnn_model = tf.keras.layers.Dense(50,
										  activation="relu",
										  name="final_dense")(cnn_model)
		
		output_layer = tf.keras.layers.Dense(self.forecast * input_shape[1],
											 name="output")(cnn_model)
		
		output_layer = tf.keras.layers.Reshape((self.forecast, input_shape[1]),
											   name="reshape")(output_layer)
		
		model = tf.keras.Model(inputs=input_layer,
							   outputs=output_layer,
							   name="lstm_model")
		
		model.compile(loss="mse",
					  optimizer=tf.keras.optimizers.Adam(),
					  metrics=[tf.keras.metrics.MeanSquaredError()])
		
		return model