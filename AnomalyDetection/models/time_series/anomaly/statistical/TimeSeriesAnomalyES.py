from copy import copy

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.time_series.anomaly.statistical.TimeSeriesAnomalyForecaster import TimeSeriesAnomalyForecaster


class TimeSeriesAnomalyES(TimeSeriesAnomalyForecaster):
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
				 es_params: dict = None):
		super().__init__(validation_split=validation_split,
						 distribution=distribution,
						 perc_quantile=perc_quantile)
		
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
			previous_points = previous.shape[0]
		else:
			previous_points = 0
		num_to_discard = self.es_params["endog"].shape[0] - previous_points
		pred = self._fitted_model.forecast(num_to_discard + x.shape[0])
		predictions = pred[num_to_discard:]
		return predictions
	
	def _model_fit(self, *args, **kwargs):
		self._fitted_model = self._model.fit(**kwargs)
	
	def _model_build(self) -> None:
		endog = self.es_params["endog"]
		num_validation = int(endog.shape[0] * self.validation_split)
		endog_training_data = endog[:-num_validation]
		new_params = copy(self.es_params)
		new_params["endog"] = endog_training_data
		
		self._model = ExponentialSmoothing(**new_params)
	
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
