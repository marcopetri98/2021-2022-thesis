from copy import copy
from typing import Tuple, Optional, Iterable

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_X_y, check_array
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from input_validation.attribute_checks import check_not_default_attributes
from models.time_series.anomaly.TimeSeriesAnomalyForecaster import TimeSeriesAnomalyForecaster
from utils.printing import print_step, print_header


class TimeSeriesAnomalyARIMA(TimeSeriesAnomalyForecaster):
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
		super().__init__(validation_split=validation_split,
						 distribution=distribution,
						 perc_quantile=perc_quantile)
		
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
		x
			Ignored by definition since ARIMA stores endogenous variables.
		"""
		super().fit(self.endog, y, verbose, fit_params, *args, **kwargs)

	def _model_predict(self, previous: np.ndarray,
					   x: np.ndarray):
		all_data = np.concatenate((previous, x))
		pred_model: ARIMAResults = self._fitted_model.apply(all_data,
															refit=False)
		prediction_results = pred_model.get_prediction(previous.shape[0])
		predictions = prediction_results.predicted_mean
		return predictions

	def _model_fit(self, *args, **kwargs):
		self._fitted_model = self._model.fit(**kwargs)

	def _model_build(self) -> None:
		num_validation = int(self.endog.shape[0] * self.validation_split)
		endog_training_data = self.endog[:-num_validation]
		
		if self.exog is not None:
			exog_training_data = self.exog[:-num_validation]
		else:
			exog_training_data = None
		
		self._model = ARIMA(endog=endog_training_data,
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

	def __check_parameters(self):
		"""Checks that the class parameters are correct.

		Returns
		-------
		None
		"""
		if self.endog is None:
			raise ValueError("It is impossible to forecast without data. Endog "
							 "must be a set of points, at least 2.")
