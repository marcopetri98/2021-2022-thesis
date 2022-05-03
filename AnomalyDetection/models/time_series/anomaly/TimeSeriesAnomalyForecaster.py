import abc
from copy import copy

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_array

from input_validation.attribute_checks import check_not_default_attributes
from models.BaseModel import BaseModel
from models.IParametric import IParametric
from models.time_series.anomaly.ITimeSeriesAnomaly import ITimeSeriesAnomaly
from utils.printing import print_step, print_header


class TimeSeriesAnomalyForecaster(ITimeSeriesAnomaly, IParametric, BaseModel):
	"""Abstract class implementing an anomaly detector based on forecasting.

	Parameters
	----------
	validation_split : float, default=0.1
		It is the percentage of training set to be used to learn the threshold
		to be able to classify points.

	distribution : str, default="gaussian"
		It is the distribution used to compute the threshold of error over which
		a point is considered an anomaly. EFFECTIVE ONLY BEFORE FITTING.

	perc_quantile : float, default=0.999
		It is the quantile to use when the distribution is fitted over the
		prediction errors of the validation set of only normal data. It will be
		the point stating that errors greater or equal to it represent an
		anomaly in data.

	Notes
	-----
	Currently, this class implements logic to train a statistical model based on
	`statsmodels <https://www.statsmodels.org/stable/api.html>` library.
	"""
	__GAUSSIAN_DIST = "gaussian"
	__TRUNC_GAUSSIAN_DIST = "truncated_gaussian"
	ACCEPTED_DISTRIBUTIONS = [__GAUSSIAN_DIST, __TRUNC_GAUSSIAN_DIST]
	
	def __init__(self, validation_split: float = 0.1,
				 distribution: str = "gaussian",
				 perc_quantile: float = 0.999):
		super().__init__()

		self.validation_split = validation_split
		self.distribution = distribution
		self.perc_quantile = perc_quantile
		self._model = None
		self._fitted_model = None
		self._threshold = - np.inf

		self.__check_parameters()

	def set_params(self, **params) -> None:
		super().set_params(**params)
		self.__check_parameters()
	
	def fit(self, x=None,
			y=None,
			verbose: bool = True,
			fit_params: dict = None,
			*args,
			**kwargs):
		"""
		Parameters
		----------
		verbose : bool, default=True
			If True, detailed printing of the process is performed. Otherwise,
			schematic printing is performed.

		fit_params : dict, default=None
			It is the dictionary of the fit arguments to pass to the wrapped
			model.
		"""
		if verbose:
			print_header("Start of the model fit")
		
		self._model_build()
		
		if verbose:
			print_step("Start to learn the parameters")
		
		if fit_params is None:
			self._model_fit()
		else:
			self._model_fit(**fit_params)
		
		if verbose:
			print_step("Parameters have been learnt")
		
		# Extract the validation set over which choose the threshold
		num_validation = int(x.shape[0] * self.validation_split)
		training_data = x[:-num_validation]
		validation_data = x[-num_validation:]
		pred_errors = self._compute_errors(training_data,
										   validation_data,
										   verbose=verbose)
		pred_errors = pred_errors.reshape((pred_errors.shape[0], 1))
		self._learn_threshold(pred_errors, verbose=verbose)
		
		if verbose:
			print_step("The model has been trained and the results are: \n",
					   self._fitted_model.summary())
			print_step("The learnt threshold is ", self._threshold)
			print_header("End of the model fit")
		
		return copy(self._fitted_model)
	
	def classify(self, X,
				 previous=None,
				 verbose: bool = True,
				 *args,
				 **kwargs) -> np.ndarray:
		"""
		Parameters
		----------
		previous : array-like of shape (n_samples, n_features)
			Data points of the time series coming before X.

		verbose : bool, default=True
			If True, detailed printing of the process is performed. Otherwise,
			synthetic printing is performed.
		"""
		check_array(X)
		check_array(previous)
		X = np.array(X)
		previous = np.array(previous)
		
		if verbose:
			print_header("Start to classify points")
		
		# Input validated in compute errors
		errors = self._compute_errors(previous, X, verbose=verbose)
		
		if verbose:
			print_step("Evaluate point on their prediction error")
		
		anomalies = np.argwhere(errors >= self._threshold)
		pred_labels = np.zeros(X.shape[0], dtype=np.intc)
		pred_labels[anomalies] = 1
		
		if verbose:
			print_header("Points' classification ended")
		
		return pred_labels
	
	def anomaly_score(self, x,
					  previous=None,
					  verbose: bool = True,
					  *args,
					  **kwargs) -> np.ndarray:
		"""
		Parameters
		----------
		previous : array-like of shape (n_samples, n_features)
			Data points of the time series coming before X.

		verbose : bool, default=True
			If True, detailed printing of the process is performed. Otherwise,
			synthetic printing is performed.
		"""
		check_array(x)
		check_array(previous)
		x = np.array(x)
		previous = np.array(previous)
		
		if verbose:
			print_header("Start to compute anomaly score of points")
		
		# Input validated in compute errors
		errors = self._compute_errors(previous, x, verbose=verbose)
		
		if verbose:
			print_step("Evaluate scores on the basis of prediction error")
		
		# For the moment, leaving the error as score could be ok
		# TODO: find a good scoring method
		
		if verbose:
			print_header("Anomaly score of points computation ended")
		
		return errors

	def regress(self, x, *args, **kwargs) -> np.ndarray:
		"""Alias for anomaly_score."""
		return self.anomaly_score(x)
	
	def predict_time_series(self, xp,
							x,
							verbose: bool = False) -> np.ndarray:
		"""Predict the future values of the time series.

		Parameters
		----------
		xp : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		verbose : bool, default=True
			If True, detailed printing of the process is performed. Otherwise,
			synthetic printing is performed.

		Returns
		-------
		labels : ndarray
			The values of the steps predicted from the time series.
		"""
		check_not_default_attributes(self, {"_fitted_model": None,
											"_model": None})
		check_array(x)
		check_array(xp)
		x = np.array(x)
		xp = np.array(xp)
		
		if verbose:
			print_step("Start to compute predictions of the test set")
		
		predictions = self._model_predict(xp, x)
		
		if verbose:
			print_step("Test set has been predicted")
		
		predictions = predictions.reshape((predictions.shape[0], 1))
		return predictions

	def _compute_errors(self, previous,
						x,
						verbose: bool = False) -> np.ndarray:
		"""Predict if a sample is an anomaly or not.

		Parameters
		----------
		previous : array-like of shape (n_samples, n_features)
			Data immediately before the values to predict.

		x : array-like of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		prediction_errors : ndarray
			Errors of the prediction.
		"""
		check_array(x)
		check_array(previous)
		x = np.array(x)
		previous = np.array(previous)

		predictions = self.predict_time_series(previous, x, verbose=verbose)

		if verbose:
			print_step("Start to compute prediction errors")

		errors = np.linalg.norm(x - predictions, axis=1)

		if verbose:
			print_step("Prediction errors have been computed")

		return errors

	def _learn_threshold(self, errors,
						 verbose: bool = False) -> None:
		"""Learn a model to evaluate the threshold for the anomaly.

		Parameters
		----------
		errors : array-like of shape (n_samples, n_features)
			Data of the prediction errors on the validation points.

		Returns
		-------
		threshold : float
			The threshold learnt from validation data.
		"""
		check_array(errors)
		errors = np.array(errors)

		if verbose:
			print_step("Start to compute the threshold on the validation data")

		match self.distribution:
			case self.__GAUSSIAN_DIST:
				# We fit a truncated gaussian to the errors (errors are scalars)
				mean = np.mean(errors)
				std = np.std(errors)
				self._threshold = norm.ppf(self.perc_quantile,
										   loc=mean,
										   scale=std)

			case self.__TRUNC_GAUSSIAN_DIST:
				# We fit a truncated gaussian to the errors (errors are scalars)
				mean = np.mean(errors)
				std = np.std(errors)
				a, b = (0 - mean) / std, (1 - mean) / std
				self._threshold = truncnorm.ppf(self.perc_quantile,
												a,
												b,
												loc=mean,
												scale=std)

		if verbose:
			print_step("Threshold has been computed")

	@abc.abstractmethod
	def _model_predict(self, previous: np.ndarray,
					   x: np.ndarray) -> np.ndarray:
		"""Predicts the values of x.

		Parameters
		----------
		previous : ndarray of shape (n_previous, n_features)
			Data immediately before the values to predict.

		x : ndarray of shape (n_samples, n_features)
			Data of the points to predict.

		Returns
		-------
		predicted : ndarray of shape (n_samples, n_features)
			The predicted values for x given the values and the preceding values.
		"""
	
	@abc.abstractmethod
	def _model_fit(self, *args, **kwargs):
		"""Fits the model to be trained.

		Parameters
		----------
		args
			Arguments to the fit of the model.

		kwargs
			Arguments to the fit of the model in keywords.

		Returns
		-------
		None
		"""
	
	@abc.abstractmethod
	def _model_build(self) -> None:
		"""Builds the model.

		Returns
		-------
		None
		"""

	def __check_parameters(self):
		"""Checks that the class parameters are correct.

		Returns
		-------
		None
		"""
		if not 0 < self.validation_split < 1:
			raise ValueError("Validation split must be in range (0,1).")
		elif self.distribution not in self.ACCEPTED_DISTRIBUTIONS:
			raise ValueError("Error distribution must be one of %s" %
							 self.ACCEPTED_DISTRIBUTIONS)
		elif not 0 < self.perc_quantile < 1:
			raise ValueError("The percentage used to compute the quantile must "
							 "lies in range (0,1)")
