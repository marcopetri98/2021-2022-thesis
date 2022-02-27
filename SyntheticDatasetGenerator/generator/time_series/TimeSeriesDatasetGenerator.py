# python libraries
import math

# external libraries
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

# python libraries
from typing import Callable

# project libraries
from generator.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from generator.exceptions.UnstableWarning import UnstableWarning
from generator.printing.DecoratedPrinter import DecoratedPrinter
from generator.time_series.Seasonality import Seasonality
from generator.time_series.Trend import Trend

class TimeSeriesDatasetGenerator(SyntheticDatasetGenerator):
	"""Concrete class used to generate time series.
	
	This class is used to create time series generator. It supports the
	generation of univariate time series with noise in input, trend, seasonality
	and point anomalies.
	"""
	# Class constants
	ALLOWED_STOCHASTIC_PROCESS = [ 	"AR",
									"MA"
									]
	ALLOWED_NOISE = [	"None",
						"Gaussian",
						"custom"
						]
	MIN_POINTS = 3
	
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
	
	# Class verbose prints
	GENERATE_PRINT = ["Dataset generation started",
				"Stochastic process generation started",
				"Stochastic process generation ended",
				"Generation of noise, trend and seasonal components started",
				"Generation of noise, trend and seasonal components ended",
				"Dataset generation ended" ]
	
	def __init__(self, supervised : bool,
				 labels : list[str]):
		super().__init__(supervised, labels)
	
	def generate(self, num_points : int,
				 dimensions : int,
				 stochastic_process : str,
				 process_params : list[float],
				 noise : list[str],
				 custom_noise : list[Callable[[], float]] = None,
				 verbose : bool = True,
				 sample_freq_seconds : float = 1.0,
				 trend : list[bool] = None,
				 seasonality : list[bool] = None,
				 trend_func : list[Trend] = None,
				 seasonality_func : list[Seasonality] = None) -> pd.DataFrame:
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
			raise NotImplementedError(self.ERROR_IMPL_PROCESS)
		elif len(process_params) == 0:
			raise ValueError(self.ERROR_PARAMS)
		elif num_points < self.MIN_POINTS:
			raise ValueError(self.ERROR_MIN_DATASET)
		elif sample_freq_seconds <= 0:
			raise ValueError(self.ERROR_FREQ_DEF)
		elif len(noise) != dimensions:
			raise ValueError(self.ERROR_NOISE_NUM)
		elif (custom_noise is not None and
			  noise.count("custom") != len(custom_noise)):
			raise ValueError(self.ERROR_NOISE_CUST_NUM)
		elif (trend is not None and
			  trend.count(True) != len(trend_func)):
			raise ValueError(self.ERROR_TREND_NUM)
		elif (seasonality is not None and
			  seasonality.count(True) != len(seasonality_func)):
			raise ValueError(self.ERROR_SEASONALITY_NUM)

		if stochastic_process == "MA" or stochastic_process == "AR":
			process_order = process_params[0]
			if len(process_params) != process_order + 3:
				raise ValueError(self.ERROR_PROC_PARAM_NUM)
			elif process_params[-1] < 0:
				raise ValueError(self.ERROR_WN_VARIANCE)

		# Given that the assumptions hold. Generate the dataset
		if verbose:
			DecoratedPrinter.print_heading(self.GENERATE_PRINT[0])
			DecoratedPrinter.print_step(self.GENERATE_PRINT[1])

		# Create the time series stochastic process and generate stationary data
		process = self.__create_process(stochastic_process, process_params)
		
		# Check process stability
		if not process.isstationary:
			warnings.warn("The process is unstable", UnstableWarning)
		
		samples_to_gen = num_points if dimensions == 1 else (num_points,
															 dimensions,
															 1)
		dataset : np.ndarray = process.generate_sample(samples_to_gen,
										math.sqrt(process_params[-1]),
										burnin=int(float(num_points) * 0.1),
										axis=0)
		if verbose:
			DecoratedPrinter.print_step(self.GENERATE_PRINT[2])
			DecoratedPrinter.print_step(self.GENERATE_PRINT[3])

		# Generate each point
		elapsed_seconds = 0
		noise_idx = 0
		trend_idx = 0
		season_idx = 0
		for point in range(num_points):
			for dim in range(dimensions):
				noise_value = 0
				trend_value = 0
				seasonal_value = 0

				if noise[dim] != "None" and noise[dim] == "custom":
					noise_value += self.__compute_noise(noise[dim],
														custom_noise[noise_idx])
					noise_idx += 1
				elif noise[dim] != "None":
					noise_value += self.__compute_noise(noise[dim])
				
				if trend is not None and trend[dim]:
					trend_value += trend_func[trend_idx]\
						.compute_trend_value(elapsed_seconds)
					trend_idx += 1
				
				if seasonality is not None and seasonality[dim]:
					seasonal_value += seasonality_func[season_idx]\
						.compute_seasonal_value(elapsed_seconds)
					season_idx += 1

				if dimensions > 1:
					dataset[point, dim, 0] += noise_value\
											  + trend_value\
											  + seasonal_value
				else:
					dataset[point] += noise_value + trend_value + seasonal_value

			elapsed_seconds += 1 / sample_freq_seconds

		# TODO: correctly format the dataset dataframe
		dataset_dataframe = pd.DataFrame(dataset)

		if verbose:
			DecoratedPrinter.print_step(self.GENERATE_PRINT[4])
			DecoratedPrinter.print_heading(self.GENERATE_PRINT[5])

		return dataset_dataframe

	@staticmethod
	def __compute_noise(noise_type : str,
						noise_func : Callable[[], float] = None) -> float:
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
	def __create_process(process_type : str,
						 params : list[float]) -> sm.tsa.ArmaProcess:
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