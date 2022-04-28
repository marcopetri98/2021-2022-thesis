from typing import Tuple, Optional, Iterable

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_X_y, check_array
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from input_validation.attribute_checks import check_not_default_attributes
from models.BaseModel import BaseModel
from models.IParametric import IParametric
from models.time_series.anomaly.ITimeSeriesAnomaly import ITimeSeriesAnomaly
from print_utils.printing import print_step, print_header


class TimeSeriesAnomalyARIMA(ITimeSeriesAnomaly, IParametric, BaseModel):
	"""ARIMA model to perform anomaly detection on time series.
	
	Parameters
	----------
	validation_split : float, default=0.1
		It is the percentage of training set to be used to learn the threshold
		to be able to classify points.
		
	distribution : str, default="gaussian"
		It is the distribution used to compute the threshold of error over which
		a point is considered an anomaly. EFFECTIVE ONLY BEFORE FITTING.
	
	Notes
	-----
	For all the other parameters that are not included in the documentation, see
	the statsmodel documentation for ARIMA models:
	https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
	"""
	__GAUSSIAN_DIST = "gaussian"
	__TRUNC_GAUSSIAN_DIST = "truncated_gaussian"
	ACCEPTED_DISTRIBUTIONS = [__GAUSSIAN_DIST, __TRUNC_GAUSSIAN_DIST]
	
	def __init__(self, validation_split: float = 0.1,
				 distribution: str = "gaussian",
				 perc_quantile: float = 0.999,
				 *,
				 endog = None,
				 exog = None,
				 order: Tuple[int, int, int] = None,
				 seasonal_order: Optional[Tuple] = None,
				 trend: str | Iterable | None = None,
				 enforce_stationarity: Optional[bool] = None,
				 enforce_invertibility: Optional[bool] = None,
				 concentrate_scale: Optional[bool] = None,
				 trend_offset: Optional[int] = None,
				 dates = None,
				 freq: Optional[str] = None,
				 missing: str = "none"):
		super().__init__()

		self.validation_split = validation_split
		self.distribution = distribution
		self.perc_quantile = perc_quantile
		self._model: ARIMA = None
		self._fitted_model: ARIMAResults = None
		self._threshold = - np.inf
		
		self.endog = np.array(endog) if endog is not None else None
		self.exog = np.array(exog) if exog is not None else None
		self.order = order
		self.seasonal_order = seasonal_order
		self.trend = trend
		self.enforce_stationarity = enforce_stationarity
		self.enforce_invertibility = enforce_invertibility
		self.concentrate_scale = concentrate_scale
		self.trend_offset = trend_offset
		self.dates = dates
		self.freq = freq
		self.missing = missing
		
		self.__check_parameters()
		
	def set_params(self, **params) -> None:
		super().set_params()
		self.__check_parameters()

	def fit(self, X=None,
			y=None,
			verbose: bool = True,
			start_params=None,
			transformed: Optional[bool] = None,
			includes_fixed: Optional[bool] = None,
			method: Optional[str] = None,
			method_kwargs: Optional[dict] = None,
			gls: Optional[bool] = None,
			gls_kwargs: Optional[dict] = None,
			cov_type: Optional[str] = None,
			cov_kwds: Optional[dict] = None,
			return_params: Optional[bool] = None,
			low_memory: Optional[bool] = None,
			*args,
			**kwargs) -> None:
		"""
		Parameters
		----------
		verbose : bool, default=True
			If True, detailed printing of the process is performed. Otherwise,
			schematic printing is performed.
		"""
		if verbose:
			print_header("Start of the model fit")
		
		self._model = self.__build_model()
		
		if verbose:
			print_step("Start to learn the parameters")
		
		self._fitted_model = self._model.fit(start_params=start_params,
											 transformed=transformed,
											 includes_fixed=includes_fixed,
											 method=method,
											 method_kwargs=method_kwargs,
											 gls=gls,
											 gls_kwargs=gls_kwargs,
											 cov_type=cov_type,
											 cov_kwds=cov_kwds,
											 return_params=return_params,
											 low_memory=low_memory)
		
		if verbose:
			print_step("Parameters have been learnt")
		
		# Extract the validation set over which choose the threshold
		num_validation = int(self.endog.shape[0] * self.validation_split)
		training_data = self.endog[:-num_validation]
		validation_data = self.endog[-num_validation:]
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
	
	def classify(self, X,
				 previous = None,
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
	
	def anomaly_score(self, X,
					  previous = None,
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
		X = np.array(X)
		previous = np.array(previous)
		
		if verbose:
			print_header("Start to compute anomaly score of points")
		
		# Input validated in compute errors
		errors = self._compute_errors(previous, X, verbose=verbose)
		
		if verbose:
			print_step("Evaluate scores on the basis of prediction error")
		
		if verbose:
			print_header("Anomaly score of points computation ended")
		
		return errors
	
	def regress(self, X, *args, **kwargs) -> np.ndarray:
		"""Alias for anomaly_score."""
		return self.anomaly_score(X)
	
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
		
		all_data = np.concatenate((xp, x))
		pred_model: ARIMAResults = self._fitted_model.apply(all_data, refit=False)
		prediction_results = pred_model.get_prediction(xp.shape[0])
		predictions = prediction_results.predicted_mean
			
		if verbose:
			print_step("Test set has been predicted")
		
		predictions = predictions.reshape((predictions.shape[0], 1))
		return predictions
		
	
	def _learn_threshold(self, x,
						 verbose: bool = False) -> None:
		"""Learn a model to evaluate the threshold for the anomaly.

		Parameters
		----------
		x : array-like of shape (n_samples, n_features)
			Data of the prediction errors on the validation points.

		Returns
		-------
		threshold : float
			The threshold learnt from validation data.
		"""
		check_array(x)
		x = np.array(x)
		
		if verbose:
			print_step("Start to compute the threshold on the validation data")
		
		match self.distribution:
			case self.__GAUSSIAN_DIST:
				# We fit a truncated gaussian to the errors (errors are scalars)
				mean = np.mean(x)
				std = np.std(x)
				self._threshold = norm.ppf(self.perc_quantile,
										   loc=mean,
										   scale=std)
			
			case self.__TRUNC_GAUSSIAN_DIST:
				# We fit a truncated gaussian to the errors (errors are scalars)
				mean = np.mean(x)
				std = np.std(x)
				a, b = (0 - mean) / std, (1 - mean) / std
				self._threshold = truncnorm.ppf(self.perc_quantile,
												a,
												b,
												loc=mean,
												scale=std)
			
		if verbose:
			print_step("Threshold has been computed")
			
	def _compute_errors(self, xp,
						x,
						verbose: bool = False) -> np.ndarray:
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
		check_array(x)
		check_array(xp)
		x = np.array(x)
		xp = np.array(xp)
		
		predictions = self.predict_time_series(xp, x, verbose=verbose)
		
		if verbose:
			print_step("Start to compute prediction errors")
			
		errors = np.linalg.norm(x - predictions, axis=1)
		
		if verbose:
			print_step("Prediction errors have been computed")
		
		return errors

	def __build_model(self) -> ARIMA:
		"""Builds the model.
		
		Returns
		-------
		arima_model: ARIMA
			The ARIMA model specified by constructor parameters.
		"""
		num_validation = int(self.endog.shape[0] * self.validation_split)
		endog_training_data = self.endog[:-num_validation]
		
		if self.exog is not None:
			exog_training_data = self.exog[:-num_validation]
		else:
			exog_training_data = None
		
		model = ARIMA(endog=endog_training_data,
					  exog=exog_training_data,
					  order=self.order,
					  seasonal_order=self.seasonal_order,
					  trend=self.trend,
					  enforce_stationarity=self.enforce_stationarity,
					  enforce_invertibility=self.enforce_invertibility,
					  concentrate_scale=self.concentrate_scale,
					  trend_offset=self.trend_offset,
					  dates=self.dates,
					  freq=self.freq,
					  missing=self.missing)
		return model
	
	def __check_parameters(self):
		"""Checks that the class parameters are correct.

		Returns
		-------
		None
		"""
		if not 0 < self.validation_split < 1:
			raise ValueError("Validation split must be in range (0,1).")
		elif self.endog is None:
			raise ValueError("It is impossible to forecast without data. Endog "
							 "must be a set of points, at least 2.")
		elif self.distribution not in self.ACCEPTED_DISTRIBUTIONS:
			raise ValueError("Error distribution must be one of %s" %
							 self.ACCEPTED_DISTRIBUTIONS)
		elif not 0 < self.perc_quantile < 1:
			raise ValueError("The percentage used to compute the quantile must "
							 "lies in range (0,1)")
