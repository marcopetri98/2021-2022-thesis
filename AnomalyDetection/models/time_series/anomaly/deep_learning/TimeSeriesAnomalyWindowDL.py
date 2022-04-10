import abc
from abc import ABC
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.callbacks import History
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y


class TimeSeriesAnomalyWindowDL(ABC):
	"""DL models using a window approach and forecast error to find anomalies.

	Parameters
	----------
	window : int, default=200
		The number of previous points to be used to predict if the following
		point is an anomaly or not.

	stride : int, default=1
		The number of steps of which the window is moved when analysing input.
		If stride=1 and the first window is [0,1,2], the second window will be
		[1,2,3].

	forecast : int, default=1
		The number of points to predict in one step.

	batch_size : int, default=32
		The batch size used for training the model. Number of points used in
		batch training.

	max_epochs : int, default=50
		The maximum number of epochs for which the model is trained.

	predict_validation : float, default=0.2
		The percentage of training samples used to perform validation for the
		DL model.

	batch_divide_training : bool, default=False
		States whether the training size must be a multiple of batch. If it is
		True and the training is not a multiple of batch, the exceeding points
		will be deleted and not considered for training.

	folder_save_path : str, default="nn_models/"
		The path of the folder to which the model must be saved and to which its
		checkpoints will be saved.

	filename : str, default="lstm"
		The name of the checkpoint file and of the model file after it has been
		trained.
	"""
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 forecast: int = 1,
				 batch_size: int = 32,
				 max_epochs: int = 50,
				 predict_validation: float = 0.2,
				 batch_divide_training: bool = False,
				 folder_save_path: str = "nn_models/",
				 filename: str = "lstm"):
		super().__init__()
		
		self.window = window
		self.stride = stride
		self.forecast = forecast
		self.batch_size = batch_size
		self.max_epochs = max_epochs
		self.predict_validation = predict_validation
		self.batch_divide_training = batch_divide_training
		self.folder_save_path = folder_save_path
		self.filename = filename
	
	@abc.abstractmethod
	def _build_x_y_sequences(self, x) -> Tuple[np.ndarray, np.ndarray]:
		"""Build the neural network inputs to perform regression.

		Parameters
		----------
		x : np.ndarray
			The training input sequence.

		Returns
		-------
		x_train : np.ndarray
			Sequences of training samples to use as training.
		y_train : np.ndarray
			Targets of each training sample to use.
		"""

		pass
	
	@abc.abstractmethod
	def _predict_future(self, xp: np.ndarray, x: np.ndarray) -> np.ndarray:
		"""Starting from the window X, it predicts the next N points.

		It predicts points in an autoregressive way using the class forecast
		dimension.

		Parameters
		----------
		xp : ndarray of shape (window, n_features)
			The window from which we have to predict the next samples.

		x : ndarray of shape (n_samples, n_features)
			The points for which we want to predict whether they are anomaly
			points or not.

		Returns
		-------
		predicted_values : ndarray of shape (points, n_features)
			The predicted values for the next points.
		"""

		pass
	
	def _learn_threshold(self, x, y) -> None:
		"""Learn a model to evaluate the threshold for the anomaly.

		Parameters
		----------
		x : array-like of shape (n_samples, n_features)
			Data of the prediction errors on the validation points.

		y : array-like of shape (n_samples, n_features)
			Data labels with shape consistent with X used to learn the decision
			boundary for the anomalies. Namely, the validation labels. 1 for
			anomalies and 0 for normal data.

		Returns
		-------
		threshold : float
			The threshold learnt from validation data.
		"""
		check_X_y(x, y)
		x = np.array(x)
		
		# We are only interested in absolute errors
		x = np.absolute(x)
		
		# We fit a truncated gaussian to the errors (errors are scalars)
		mean = np.mean(x)
		std = np.std(x)
		a, b = (0 - mean) / std, (1 - mean) / std
		self.threshold_ = truncnorm.ppf(0.99, a, b, loc=mean, scale=std)
	
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
		
		# List of the histories for the training on the various data
		histories = []
		input_shape = (self.window, x.shape[1])
		self.model_ = self._learning_create_model(input_shape)
		xs = []
		
		for slice_ in training_idx:
			xs.append(x[slice_])
		
		# Perform training on each training slice
		for data in xs:
			if self.window > data.shape[0]:
				raise ValueError("Window cannot be larger than data size.")
			elif data.shape[1] > 1:
				raise ValueError("Only univariate time series is currently "
								 "supported.")
			elif (data.shape[0] - self.window) % self.stride != 0:
				raise ValueError("Data.shape[0] - window must be a multiple of "
								 "stride to learn from it.")
			
			if self.batch_divide_training:
				if (data.shape[0] - self.window) % self.batch_size != 0:
					raise ValueError("Data.shape[0] - window must be a multiple"
									 " of batch to build the spatial data. I.e.,"
									 "(Data.shape[0] - window)%batch_size == 0")
			
			# Build the train sequences from the given input
			x_train, y_train = self._build_x_y_sequences(data)
			split = self.predict_validation
			points = int(x_train.shape[0] * split)
			
			# If the model is stateful, reshape correctly
			if self.batch_divide_training:
				points = int(x_train.shape[0] / self.batch_size) * self.batch_size
				x_train, y_train = x_train[:points], y_train[:points]
				
				points = int(x_train.shape[0] * split / self.batch_size) * self.batch_size
			
			# Build training and validation sets
			x_val = x_train[-points:]
			y_val = y_train[-points:]
			x_train = x_train[:-points]
			y_train = y_train[:-points]
			
			# Fit the model on this slice
			self.model_.summary()
			checkpoint_path = self.folder_save_path + "/checkpoint/" + self.filename + ".h5"
			history = self.model_.fit(
				x=x_train,
				y=y_train,
				batch_size=self.batch_size,
				epochs=self.max_epochs,
				validation_data=(x_val, y_val),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor="val_loss",
													 patience=30,
													 mode="min",
													 restore_best_weights=True),
					tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
														 factor=0.1,
														 patience=20,
														 mode="min"),
					tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
													   monitor="val_loss")
				]
			)
			
			# Save history and reset state and metrics before training on the
			# next time series values.
			histories.append(history)
			self.model_.reset_states()
			self.model_.reset_metrics()
		
		# Create the prediction model
		trained_model = self.model_
		self.model_ = self._prediction_create_model(input_shape)
		self.model_.set_weights(trained_model.get_weights())
		self.model_.save(self.folder_save_path + self.filename,
						 save_format="h5")
		
		# Compute the predictions of the model and build supervised values
		valid_predictions = np.array([[]])
		valid_true_labels = np.array([[]])
		for slice_ in validation_idx:
			errors = self._compute_errors(x[:slice_.stop], x[slice_])
			
			# Numpy works with a shape of (n_samples, n_features)
			errors = errors.reshape((errors.shape[0], 1))
			if valid_predictions.size == 0:
				valid_predictions = errors.copy()
				valid_true_labels = y[slice_].copy()
			else:
				valid_predictions = np.concatenate((valid_predictions, errors))
				valid_true_labels = np.concatenate((valid_true_labels, y[slice_]))
		
		# Learn the anomaly threshold
		self._learn_threshold(valid_predictions, valid_true_labels)
		threshold_file = self.folder_save_path + self.filename + ".threshold"
		np.save(threshold_file, np.array([self.threshold_]))
		
		return histories
	
	def _compute_errors(self, xp, x) -> np.ndarray:
		"""Predict if a sample is an anomaly or not.

		Parameters
		----------
		xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		prediction_errors : ndarray
			Errors of the prediction.
		"""
		predictions = self.predict_time_series(xp, x)
		return np.linalg.norm(x - predictions, axis=1)
	
	def anomaly_score(self, xp, x) -> np.ndarray:
		"""Predict if a sample is an anomaly or not.

		Parameters
		----------
		xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		labels : ndarray
			The anomaly scores of each point. Greater than 0 means anomaly and
			less than 0 is normal. The bigger the more abnormal.
		"""
		# Input validated in compute errors
		errors = self._compute_errors(xp, x)
		errors = np.abs(errors)
		
		scores = errors.reshape((errors.shape[0], 1))
		scores = MinMaxScaler().fit_transform(scores)
		scores = scores.reshape(scores.shape[0])
		
		return scores
	
	@abc.abstractmethod
	def predict_time_series(self, xp, x) -> np.ndarray:
		"""Predict the future values of the time series.

		Parameters
		----------
		xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		labels : ndarray
			The values of the steps predicted from the time series.
		"""
		pass
	
	def predict(self, xp, x) -> np.ndarray:
		"""Predict if a sample is an anomaly or not.

		Parameters
		----------
		xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		labels : ndarray
			The labels for each point in X. 1 for an anomaly and 0 for a normal
			point.
		"""
		# Input validated in compute errors
		errors = self._compute_errors(xp, x)
		
		anomalies = np.argwhere(errors >= self.threshold_)
		pred_labels = np.zeros(x.shape[0])
		pred_labels[anomalies] = 1
		
		return pred_labels
	
	def load_model(self, file_path: str) -> None:
		self.model_ = tf.keras.models.load_model(file_path)
		threshold_file = file_path + ".threshold.npy"
		threshold_array = np.load(threshold_file)
		self.threshold_ = threshold_array[0]
	
	@abc.abstractmethod
	def _prediction_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		"""Creates the model to be used for predictions.

		Generally, it is identical to the learning model. However, there are
		some cases in which we want to change the batch size of the model from
		training to prediction. For instance, we would like to train the model
		using a mini-batch training while we want to produce predictions for one
		point at a time.

		Parameters
		----------
		input_shape : Tuple
			Represents the shape of the input to the prediction model.

		Returns
		-------
		model : tf.keras.Model
			The model for the prediction.
		"""
		pass
	
	@abc.abstractmethod
	def _learning_create_model(self, input_shape: Tuple) -> tf.keras.Model:
		"""Creates the model to be used for learning.

		It is the network we want to train on anomaly detection tasks.

		Parameters
		----------
		input_shape : Tuple
			Represents the shape of the input to the prediction model.

		Returns
		-------
		model : tf.keras.Model
			The model for the prediction.
		"""
		pass
