# python libraries
from __future__ import annotations
from typing import Callable
from dataclasses import dataclass

import math
import datetime
import warnings

# external libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

# project libraries
from generator.SyntheticGenerator import SyntheticGenerator
from generator.exceptions.UnstableWarning import UnstableWarning
from generator.printing.DecoratedPrinter import DecoratedPrinter
from generator.time_series.Seasonality import Seasonality
from generator.time_series.Trend import Trend


@dataclass(frozen=True)
class TimeSeriesPrints(object):
	# Class warnings
	WARNING_UNSTABLE = "The process is unstable"

	# Class errors
	ERROR_IMPL_PROCESS = "This process is not implemented"
	ERROR_IMPL_NOISE = "This noise is not implemented"
	ERROR_PARAMS = "You must give process parameters"
	ERROR_MIN_DATASET = "A dataset must be of at least 2 points"
	ERROR_FREQ_DEF = "Frequency must be greater than 0"
	ERROR_NOISE_NUM = "You must define noise type for each dimension"
	ERROR_NOISE_CUST_NUM = "Wrong number of passed custom noises"
	ERROR_TREND_NUM = "Wrong number of passed trend functions"
	ERROR_SEASONALITY_NUM = "Wrong number of passed seasonality functions"
	ERROR_PROC_PARAM_NUM = "Wrong number of process params"
	ERROR_WN_VARIANCE = "White noise variance must be >= 0"
	ERROR_NUM_COLUMNS = "The number of columns must be identical to dimensions"
	ERROR_TIMESTAMP = "Timestamp must be greater or equal 0"

	# Class verbose prints
	GENERATE_PRINT = ["Dataset generation started",
					  "Stochastic process generation started",
					  "Stochastic process generation ended",
					  "Generation of noise, trend and seasonal components started",
					  "Generation of noise, trend and seasonal components ended",
					  "Dataset generation ended"]


class TimeSeriesGenerator(SyntheticGenerator):
	"""Concrete class used to generate time series.

	This class is used to create time series generator. It supports the
	generation of univariate time series with noise in input, trend, seasonality
	and point anomalies.

	Attributes
	----------

	* dataset_timestamps: the data timestamps for time series values.
	* dataset: the numpy ndarray representing the dataset.
	* dataset_frame: the pandas dataframe representing the dataset.
	* supervised: a boolean value representing if the dataset is supervised or
	not.
	* labels: labels to be used for the dataset.
	"""
	# Class constants
	ALLOWED_STOCHASTIC_PROCESS = ["AR",
								  "MA"]
	ALLOWED_NOISE = ["None",
					 "Gaussian",
					 "custom"]
	MIN_POINTS = 3

	def __init__(self, supervised: bool,
				 labels: list[str]):
		super().__init__(supervised, labels)
		self.dataset_timestamps = None
		self.dataset = None
		self.dataset_frame = None

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
				 start_timestamp: float = datetime.datetime.now().timestamp()) -> TimeSeriesGenerator:
		"""Generate a dataset with the given parameters.

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

		Notes
		-----
		More detailed description of lag polynomial representation can be found
		on statsmodels documentation of ArmaProcess.
		"""
		# Check that assumptions hold
		# TODO: implement also multivariate
		if dimensions < 1:
			raise NotImplementedError("Only univariate time series generation"
									  " has been implemented. Dimensions = 1")
		elif stochastic_process not in self.ALLOWED_STOCHASTIC_PROCESS:
			raise NotImplementedError(TimeSeriesPrints.ERROR_IMPL_PROCESS)
		elif len(process_params) == 0:
			raise ValueError(TimeSeriesPrints.ERROR_PARAMS)
		elif num_points < self.MIN_POINTS:
			raise ValueError(TimeSeriesPrints.ERROR_MIN_DATASET)
		elif sample_freq_seconds <= 0:
			raise ValueError(TimeSeriesPrints.ERROR_FREQ_DEF)
		elif len(noise) != dimensions:
			raise ValueError(TimeSeriesPrints.ERROR_NOISE_NUM)
		elif (custom_noise is not None and
			  noise.count("custom") != len(custom_noise)):
			raise ValueError(TimeSeriesPrints.ERROR_NOISE_CUST_NUM)
		elif (trend is not None and
			  trend.count(True) != len(trend_func)):
			raise ValueError(TimeSeriesPrints.ERROR_TREND_NUM)
		elif (seasonality is not None and
			  seasonality.count(True) != len(seasonality_func)):
			raise ValueError(TimeSeriesPrints.ERROR_SEASONALITY_NUM)
		elif (columns_names is not None and
			  len(columns_names) != dimensions):
			raise ValueError(TimeSeriesPrints.ERROR_NUM_COLUMNS)
		elif start_timestamp < 0:
			raise ValueError(TimeSeriesPrints.ERROR_TIMESTAMP)

		if stochastic_process == "MA" or stochastic_process == "AR":
			process_order = process_params[0]
			if len(process_params) != process_order + 3:
				raise ValueError(TimeSeriesPrints.ERROR_PROC_PARAM_NUM)
			elif process_params[-1] < 0:
				raise ValueError(TimeSeriesPrints.ERROR_WN_VARIANCE)

		# Given that the assumptions hold. Generate the dataset
		if verbose:
			DecoratedPrinter.print_heading(TimeSeriesPrints.GENERATE_PRINT[0])
			DecoratedPrinter.print_step(TimeSeriesPrints.GENERATE_PRINT[1])

		# Create the time series stochastic process and generate stationary data
		process = self.__create_process(stochastic_process, process_params)

		# Check process stability
		if not process.isstationary:
			warnings.warn(TimeSeriesPrints.WARNING_UNSTABLE, UnstableWarning)

		samples_to_gen = num_points if dimensions == 1 else (num_points,
															 dimensions,
															 1)
		dataset: np.ndarray = process.generate_sample(samples_to_gen,
													   math.sqrt(process_params[-1]),
													   burnin=int(float(num_points) * 0.1),
													   axis=0)
		if verbose:
			DecoratedPrinter.print_step(TimeSeriesPrints.GENERATE_PRINT[2])
			DecoratedPrinter.print_step(TimeSeriesPrints.GENERATE_PRINT[3])

		# Generate each point
		elapsed_seconds = 0
		noise_bool = [True if x == "custom" else False for x in noise]
		for point in range(num_points):
			for dim in range(dimensions):
				noise_value = 0
				trend_value = 0
				seasonal_value = 0

				if noise[dim] != "None" and noise[dim] == "custom":
					idx = sum(noise_bool[0:dim + 1]) - 1
					noise_value += self.__compute_noise(noise[dim],
														custom_noise[idx])
				elif noise[dim] != "None":
					noise_value += self.__compute_noise(noise[dim])

				if trend is not None and trend[dim]:
					idx = sum(trend[0:dim + 1]) - 1
					trend_value += trend_func[idx] \
						.compute_trend_value(elapsed_seconds)

				if seasonality is not None and seasonality[dim]:
					idx = sum(seasonality[0:dim + 1]) - 1
					seasonal_value += seasonality_func[idx] \
						.compute_seasonal_value(elapsed_seconds)

				if dimensions > 1:
					dataset[point, dim, 0] += noise_value \
											  + trend_value \
											  + seasonal_value
				else:
					dataset[point] += noise_value + trend_value + seasonal_value

			elapsed_seconds += 1 / sample_freq_seconds

		# Save the numpy dataset
		self.dataset = dataset

		# Save the pandas dataframe version of the dataset
		self.__save_dataframe(dataset,
							  num_points,
							  sample_freq_seconds,
							  columns_names,
							  start_timestamp)

		if verbose:
			DecoratedPrinter.print_step(TimeSeriesPrints.GENERATE_PRINT[4])
			DecoratedPrinter.print_heading(TimeSeriesPrints.GENERATE_PRINT[5])

		return self

	def __save_dataframe(self, dataset: np.ndarray,
						 num_points: int,
						 sample_freq_seconds: float,
						 columns_names: list[str],
						 start_timestamp: float) -> None:
		"""Computes and saves the dataframe version of the dataset.

		Parameters
		----------

		* dataset: the numpy array version of the dataset.
		"""
		start = float(math.floor(start_timestamp))
		secs_per_sample = float(math.floor(1 / sample_freq_seconds))
		last_timestamp = start + (num_points - 1) * secs_per_sample
		timestamps = np.linspace(start,
								 last_timestamp,
								 num_points)
		self.dataset_timestamps = [x for x in timestamps]
		timestamps = [pd.Timestamp(x, unit="s") for x in timestamps]
		index = pd.DatetimeIndex(timestamps)
		if columns_names is None:
			self.dataset_frame = pd.DataFrame(dataset,
											  index,
											  dtype=np.double)
		else:
			self.dataset_frame = pd.DataFrame(dataset,
											  index,
											  columns_names,
											  dtype=np.double)

	@staticmethod
	def __compute_noise(noise_type: str,
						noise_func: Callable[[], float] = None) -> float:
		"""Computes the noise value given the noise type.

		Parameters
		----------

		* noise_type: a string representing the type of noise.
		"""
		noise = 0

		match noise_type:
			case "Gaussian":
				noise += np.random.standard_normal()

			case "custom":
				noise += noise_func()

		return noise

	@staticmethod
	def __create_process(process_type: str,
						 params: list[float]) -> sm.tsa.ArmaProcess:
		"""Given the type of process and the parameters, it creates it.

		Parameters
		----------

		* process_type: the type of process between ones allowed.
		* params: the parameters of the process.

		Returns
		-------

		* sm.tsa.ArmaProcess representing the ARMA process as given by the
		parameters.
		"""
		process = None

		match process_type:
			case "AR":
				parameters = np.array(params[1:-1])
				process = sm.tsa.ArmaProcess(parameters, [1])

			case "MA":
				parameters = np.array(params[1:-1])
				process = sm.tsa.ArmaProcess([1], parameters)

		return process
