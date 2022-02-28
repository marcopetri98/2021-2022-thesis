# python libraries
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import datetime

# project libraries
from generator.time_series.AnomalyDistribution import AnomalyDistribution
from generator.time_series.Seasonality import Seasonality
from generator.time_series.TimeSeriesGenerator import TimeSeriesGenerator
from generator.time_series.Trend import Trend


@dataclass(frozen=True)
class AnomalyTimeSeriesPrints(object):
	ERROR_IMPL_ANOMALY = "Anomaly type not implemented"
	ERROR_TOO_MANY_ANOMALIES = "Anomalies must be in the interval (0, 0.2]"


class AnomalyTimeSeriesGenerator(TimeSeriesGenerator):
	"""Class defining anomaly time series dataset generation

	Attributes
	----------

	* dataset_timestamps: the data timestamps for time series values.
	* dataset: the numpy ndarray representing the dataset.
	* dataset_frame: the pandas dataframe representing the dataset.
	* supervised: a boolean value representing if the dataset is supervised or
	not.
	* labels: labels to be used for the dataset.
	* anomalies: a string representing the anomalies present in the dataset.
	* anomalies_perc: percentage of anomalies in the dataset.
	"""
	ALLOWED_ANOMALIES = ["point",
						 "collective"]

	def __init__(self, supervised: bool,
				 labels: list[str]):
		super().__init__(supervised, labels)
		self.anomalies = None
		self.anomalies_perc = None

	def generate(self, num_points: int,
				 dimensions: int,
				 stochastic_process: str,
				 process_params: list[float],
				 noise: list[str],
				 custom_noise: list[Callable[[], float]] = None,
				 verbose: bool = True,
				 sample_freq_seconds: float = 1.0,
				 trend: list[bool] = None,
				 seasonality: list[bool] = None,
				 trend_func: list[Trend] = None,
				 seasonality_func: list[Seasonality] = None,
				 columns_names: list[str] = None,
				 start_timestamp: float = datetime.datetime.now().timestamp()) -> AnomalyTimeSeriesGenerator:
		super().generate(num_points,
						 dimensions,
						 stochastic_process,
						 process_params,
						 noise,
						 custom_noise,
						 verbose,
						 sample_freq_seconds,
						 trend,
						 seasonality,
						 trend_func,
						 seasonality_func,
						 columns_names,
						 start_timestamp)
		return self

	def add_anomalies(self, anomalies: str,
					  anomaly_dist: AnomalyDistribution,
					  anomalies_perc: float = 0.01) -> AnomalyTimeSeriesGenerator:
		"""Add anomalies to the generated time series.

		Parameters
		----------

		* anomalies: a string representing the anomalies present in the dataset.
		* anomaly_dist: the distribution of anomalies on the time series.
		* anomalies_perc: percentage of anomalies in the dataset.
		"""
		if anomalies_perc <= 0 or anomalies_perc >= 1:
			raise ValueError(AnomalyTimeSeriesPrints.ERROR_TOO_MANY_ANOMALIES)
		elif anomalies not in self.ALLOWED_ANOMALIES:
			raise ValueError(AnomalyTimeSeriesPrints.ERROR_IMPL_ANOMALY)



		return self
