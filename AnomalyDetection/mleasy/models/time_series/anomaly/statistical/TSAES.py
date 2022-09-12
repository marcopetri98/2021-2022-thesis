from copy import copy

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from mleasy.models.time_series.anomaly.statistical.TSAForecaster import TSAForecaster


class TSAES(TSAForecaster):
	"""ES model to perform anomaly detection on time series.
	
	When points are predicted using SES, it is important to know that the points
	to be predicted must be the whole sequence immediately after the training.
	It is not possible to predicted from the 50th point to the 100th with the
	current implementation.
	
	Notes
	-----
	For all the other parameters that are not included in the documentation, see
	the `statsmodels <https://www.statsmodels.org/stable/api.html>`
	documentation for `ARIMA models <https://www.statsmodels.org/stable/generate
	d/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html>`."""
	
	def __init__(self, validation_split: float = 0.1,
				 distribution: str = "gaussian",
				 perc_quantile: float = 0.999,
				 scoring: str = "difference",
				 es_params: dict = None):
		super().__init__(validation_split=validation_split,
						 distribution=distribution,
						 perc_quantile=perc_quantile,
						 scoring=scoring)
		
		# Estimated parameters
		self.alpha = None
		self.beta = None
		self.gamma = None
		self.phi = None
		self.l0 = None
		self.b0 = None
		self.s0 = None
		
		# Type of training
		self.trend = None
		self.damped_trend = None
		self.seasonal = None
		self.seasonal_periods = None
		
		self.es_params = es_params
		
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
		return super().fit(self.es_params["endog"],
						   y,
						   verbose,
						   fit_params,
						   *args,
						   **kwargs)
	
	def _model_predict(self, previous: np.ndarray,
					   x: np.ndarray):
		if previous.shape != tuple(()):
			all_data = np.concatenate((previous, x))
		else:
			all_data = x
			
		self._model = ExponentialSmoothing(all_data,
										   initialization_method="known",
										   trend=self.trend,
										   damped_trend=self.damped_trend,
										   seasonal=self.seasonal,
										   seasonal_periods=self.seasonal_periods,
										   initial_level=self.l0,
										   initial_trend=self.b0,
										   initial_seasonal=self.s0)
		self._fitted_model = self._model.fit(smoothing_level=self.alpha,
											 smoothing_trend=self.beta,
											 smoothing_seasonal=self.gamma,
											 damping_trend=self.phi,
											 optimized=False)
		
		predictions = self._fitted_model.predict(start=all_data.shape[0] - x.shape[0],
												 end=all_data.shape[0] - 1)
	
		return predictions
	
	def _model_fit(self, *args, **kwargs):
		self._fitted_model = self._model.fit(**kwargs)
		
		learnt_params = self._fitted_model.params_formatted
		index_of_seasonal = 3
		
		if self.trend is not None:
			self.beta = learnt_params.loc["smoothing_trend"]["param"]
			self.b0 = learnt_params.loc["initial_trend"]["param"]
			index_of_seasonal += 2
		if self.damped_trend is not None:
			self.phi = learnt_params.loc["damping_trend"]["param"]
			index_of_seasonal += 1
		if self.seasonal is not None:
			self.gamma = learnt_params.loc["smoothing_seasonal"]["param"]
			self.s0 = np.array(learnt_params["param"].iloc[index_of_seasonal:])
		
		self.alpha = learnt_params.loc["smoothing_level"]["param"]
		self.l0 = learnt_params.loc["initial_level"]["param"]
	
	def _model_build(self, inplace: bool=True) -> None | object:
		endog = self.es_params["endog"]
		num_validation = int(endog.shape[0] * self.validation_split)
		endog_training_data = endog[:-num_validation]
		new_params = copy(self.es_params)
		new_params["endog"] = endog_training_data
		
		if inplace:
			if "trend" in self.es_params.keys():
				self.trend = self.es_params["trend"]
			if "damped_trend" in self.es_params.keys():
				self.damped_trend = self.es_params["damped_trend"]
			if "seasonal" in self.es_params.keys():
				self.seasonal = self.es_params["seasonal"]
			if "seasonal_periods" in self.es_params.keys():
				self.seasonal_periods = self.es_params["seasonal_periods"]
			
			self._model = ExponentialSmoothing(**new_params)
		else:
			return ExponentialSmoothing(**new_params)
	
	def __check_parameters(self):
		"""Checks that the class parameters are correct.

		Returns
		-------
		None
		"""
		if "endog" in self.es_params.keys():
			if self.es_params["endog"] is None:
				raise ValueError("It is impossible to forecast without data. "
								 "Endog must be a set of points, at least 2.")
