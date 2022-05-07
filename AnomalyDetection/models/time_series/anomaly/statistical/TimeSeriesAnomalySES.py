from models.time_series.anomaly.statistical.TimeSeriesAnomalyES import TimeSeriesAnomalyES


class TimeSeriesAnomalySES(TimeSeriesAnomalyES):
	"""SES model to perform anomaly detection on time series.
	
	The SES implemented by statsmodels is the Holt-Winters definition of Simple
	Exponential Smoothing, which is a specific case of Exponential smoothing.
	For more details check the `statsmodels <https://www.statsmodels.org/stable/
	api.html>` implementation.
	
	When points are predicted using SES, it is important to know that the points
	to be predicted must be the whole sequence immediately after the training.
	It is not possible to predicted from the 50th point to the 100th with the
	current implementation.
	
	This model is a sub-class of :class:`~models.time_series.anomaly.statistical
	.TimeSeriesAnomalyES` since SES is conceptually the same thing of ES where
	we have neither trend nor seasonality.
	
	Notes
	-----
	For all the other parameters that are not included in the documentation, see
	the `statsmodels <https://www.statsmodels.org/stable/api.html>`
	documentation for `SES models <https://www.statsmodels.org/stable/generated/
	statsmodels.tsa.holtwinters.SimpleExpSmoothing.html>`."""

	def __init__(self, validation_split: float = 0.1,
				 distribution: str = "gaussian",
				 perc_quantile: float = 0.999,
				 scoring: str = "difference",
				 ses_params: dict = None):
		super().__init__(validation_split=validation_split,
						 distribution=distribution,
						 perc_quantile=perc_quantile,
						 scoring=scoring,
						 es_params=ses_params)
