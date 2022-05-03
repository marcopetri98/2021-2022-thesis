from copy import copy

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from models.time_series.anomaly.TimeSeriesAnomalyForecaster import TimeSeriesAnomalyForecaster


class TimeSeriesAnomalySES(TimeSeriesAnomalyForecaster):
	"""SES model to perform anomaly detection on time series.
	
	The SES implemented by statsmodels is the Holt-Winters definition of Simple
	Exponential Smoothing, which is a specific case of Exponential smoothing.
	For more details check the `statsmodels <https://www.statsmodels.org/stable/
	api.html>` implementation.
	
	When points are predicted using SES, it is important to know that the points
	to be predicted must be the whole sequence immediately after the training.
	It is not possible to predicted from the 50th point to the 100th with the
	current implementation.
	
	Notes
	-----
	For all the other parameters that are not included in the documentation, see
	the `statsmodels <https://www.statsmodels.org/stable/api.html>`
	documentation for `SES models <https://www.statsmodels.org/stable/generated/
	statsmodels.tsa.holtwinters.SimpleExpSmoothing.html>`."""

	def __init__(self, validation_split: float = 0.1,
				 distribution: str = "gaussian",
				 perc_quantile: float = 0.999,
				 ses_params: dict = None):
		super().__init__(validation_split=validation_split,
						 distribution=distribution,
						 perc_quantile=perc_quantile)
		
		self.ses_params = ses_params

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
		return super().fit(self.ses_params["endog"],
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
		num_to_discard = self.ses_params["endog"].shape[0] - previous_points
		pred = self._fitted_model.forecast(num_to_discard + x.shape[0])
		predictions = pred[num_to_discard:]
		return predictions
	
	def _model_fit(self, *args, **kwargs):
		self._fitted_model = self._model.fit(**kwargs)
	
	def _model_build(self) -> None:
		endog = self.ses_params["endog"]
		num_validation = int(endog.shape[0] * self.validation_split)
		endog_training_data = endog[:-num_validation]
		new_params = copy(self.ses_params)
		new_params["endog"] = endog_training_data
		
		self._model = SimpleExpSmoothing(**new_params)
	
	def __check_parameters(self):
		"""Checks that the class parameters are correct.

		Returns
		-------
		None
		"""
		if "endog" in self.ses_params.keys():
			if self.ses_params["endog"] is None:
				raise ValueError("It is impossible to forecast without data. "
								 "Endog must be a set of points, at least 2.")
