# python libraries
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Union

import random
import datetime

# external libraries
import numpy as np
import pandas as pd

# project libraries
from generator.printing.DecoratedPrinter import DecoratedPrinter
from generator.time_series.Anomaly import Anomaly
from generator.time_series.AnomalyDistribution import AnomalyDistribution
from generator.time_series.Seasonality import Seasonality
from generator.time_series.TimeSeriesGenerator import TimeSeriesGenerator
from generator.time_series.Trend import Trend


@dataclass(frozen=True)
class AnomalyTimeSeriesPrints(object):
	ERROR_IMPL_ANOMALY = "Anomaly type not implemented"
	ERROR_TOO_MANY_ANOMALIES = "Anomalies must be in the interval (0, 0.2]"
	ERROR_TOO_SHORT = "A collective anomaly must have at least 2 points"
	ERROR_WRONG_PARAMS = "The parameters are wrong for the specified anomaly"
	
	GENERATE = ["Anomaly detection dataset generation started",
				"Anomaly detection dataset generation ended",
				"Generation of the time series started",
				"Generate anomalies' position",
				"Add the anomalies on the dataset",
				"Creating the ground truth to populate",
				"Saving the dataframe of the dataset"]


class AnomalyTimeSeriesGenerator(TimeSeriesGenerator):
	"""Class defining anomaly time series dataset generation

	Attributes
	----------

	* dataset_timestamps: the data timestamps for time series values.
	* dataset: the numpy ndarray representing the dataset.
	* dataset_frame: the pandas dataframe representing the dataset.
	* supervised: a boolean value representing if the dataset is supervised or
	not.
	* labels: inherited but unused. Labels to be used for the dataset.
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
				 start_timestamp: float = datetime.datetime.now().timestamp(),
				 anomalies: Union[list[Anomaly], list[list[Anomaly]]] = None,
				 anomaly_dist: AnomalyDistribution = None,
				 anomalies_perc: float = 0.01,
				 *args, **kwargs) -> AnomalyTimeSeriesGenerator:
		"""Generate an anomaly detection time series dataset.

		Parameters
		----------

		* num_points: an integer number representing the dimensionality of the
		dataset.
		* dimensions: number of dimensions of the dataset. CURRENTLY, ONLY
		SUPPORTS 1.
		* stochastic_process: a string either "AR" or "MA" identifying the
		process from which the time series is generated.
		* process_params: a list of the params defining the stochastic process
		using the lag polynomial representation including the zero lag. In case
		or "AR" or "MA" the first parameter is the order, the following are the
		coefficients (lag polynomial representation) and the last parameter
		is the variance of the white noise. See notes for details.
		* noise: list of noise types for each dimension. Possible noise types
		are "None", "Gaussian" and "custom". The dimension of the list must be
		equal to the number of dimensions of the dataset. Moreover, at index i
		there is the noise for dimension i. NOTE that this has nothing to deal
		with the noise of the time series. It is an additional noise to the
		dimension. For instance, noise we can have for sensors and other.
		* custom_noise: [optional] a list with dimensions equal to the number of
		dimensions for which noise is "custom". The element at index i
		represents the implementation of the (i+1)th custom noise. E.g., if we
		have noise = ["Gaussian", "custom", "Gaussian", "custom"]. Then the
		element custom_noise[0] for custom_noise is the implementation of
		noise[1] while custom_noise[1] is the implementation of noise[3].
		* verbose: if true, detailed prints will be performed. Otherwise
		concise prints will be done.
		* sample_freq_seconds: the number of samples per second.
		* trend: if true at index i, dimension i has a trend.
		* seasonality: if true at index i, dimension i has a seasonality.
		* trend_func: list of dimension equal to number of True values in trend.
		The Trend at trend_func[i] represents the trend of the (i+1)th True
		value in trend.
		* seasonality_func: list of dimension equal to the number of True values
		in seasonality. The Seasonality at seasonality_func[i] represents the
		seasonality of the (i+1)th True value in seasonality.
		* columns_names: list of the column names to use for the time series.
		If given, len(columns_names) must be identical to dim.
		* start_timestamp: the timestamp at which to start the time series.
		* anomalies: a list of all the possible anomalies we can encounter in
		the dataset being generated. They can even be of different types one
		from another.
		* anomaly_dist: the distribution of anomalies on the time series.
		* anomalies_perc: percentage of anomalies in the dataset.

		Notes
		-----
		More detailed description of lag polynomial representation can be found
		on statsmodels documentation of ArmaProcess.

		anomalies and anomaly_dist are required, although they are non-default
		parameters to be stick with base class signature.
		"""
		if verbose:
			DecoratedPrinter.print_heading(AnomalyTimeSeriesPrints.GENERATE[0])
			DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[2])
			
		super().generate(num_points,
						 dimensions,
						 stochastic_process,
						 process_params,
						 noise,
						 custom_noise,
						 False,
						 sample_freq_seconds,
						 trend,
						 seasonality,
						 trend_func,
						 seasonality_func,
						 columns_names,
						 start_timestamp)
		if anomalies is None or anomaly_dist is None:
			raise ValueError("anomalies and anomaly_dist must be given")
		elif anomalies_perc <= 0 or anomalies_perc >= 1:
			raise ValueError(AnomalyTimeSeriesPrints.ERROR_TOO_MANY_ANOMALIES)
		elif isinstance(anomalies[0], list):
			raise NotImplementedError("Only univariate implemented")

		if columns_names is not None and self.supervised:
			columns_names = columns_names + ["target_"+c for c in columns_names]

		num_points = self.dataset.shape[0]
		num_anomalies = int(num_points * anomalies_perc)

		if verbose:
			DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[3])
		# TODO: implement also multivariate
		# Generate indexes at which anomalies are found
		indexes = anomaly_dist.generate_anomaly_positions(num_points,
														  num_anomalies)

		# Generate the anomalies
		anomalies_generated = 0

		if self.supervised:
			if verbose:
				DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[5])
			self.ground_truth = np.ndarray(self.dataset.shape, dtype=object)
			self.ground_truth.fill("normal")

		if verbose:
			DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[4])

		for idx in indexes:
			if anomalies_generated >= num_anomalies:
				break

			anomaly = random.choice(anomalies)
			generated = anomaly.compute_anomaly(self.dataset, idx)
			if isinstance(generated, list):
				# In case collective is generated at the end, we do clipping
				clipping = len(self.dataset[idx:idx + len(generated)])
				self.dataset[idx:idx + len(generated)] = generated[0:clipping]
				if self.supervised:
					self.ground_truth[idx:idx + len(generated)] = "anomaly"
			else:
				self.dataset[idx] = generated
				if self.supervised:
					self.ground_truth[idx] = "anomaly"

		if verbose:
			DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[6])
		
		self.__save_anomaly(num_points,
							sample_freq_seconds,
							dimensions,
							columns_names,
							start_timestamp)
		if verbose:
			DecoratedPrinter.print_heading(AnomalyTimeSeriesPrints.GENERATE[1])

		return self

	def __save_anomaly(self, num_points: int,
					   sample_freq_seconds: float,
					   dimensions: int,
					   columns_names: list[str],
					   start_timestamp: float) -> None:
		"""Saves the anomaly dataframe version of the dataset"""
		timestamps = [pd.Timestamp(x, unit="s") for x in self.dataset_timestamps]
		index = pd.DatetimeIndex(timestamps)

		if not self.supervised:
			if columns_names is None:
				self.dataset_frame = pd.DataFrame(self.dataset,
												  index,
												  dtype=np.double)
			else:
				self.dataset_frame = pd.DataFrame(self.dataset,
												  index,
												  columns_names,
												  dtype=np.double)
		else:
			new_shape = (num_points, dimensions * 2)
			complete_ds: np.ndarray = np.ndarray(new_shape, dtype=object)
			complete_ds[:, 0:dimensions] = np.reshape(self.dataset, (num_points, 1))
			complete_ds[:, dimensions:2 * dimensions] = np.reshape(self.ground_truth, (num_points, 1))
			self.dataset = complete_ds
				
			if columns_names is None:
				self.dataset_frame = pd.DataFrame(complete_ds,
												  index,
												  dtype=object)
			else:
				self.dataset_frame = pd.DataFrame(complete_ds,
												  index,
												  columns_names,
												  dtype=object)