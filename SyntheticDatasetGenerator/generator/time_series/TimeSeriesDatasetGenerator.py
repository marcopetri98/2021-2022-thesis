import pandas as pd

from typing import Callable
from generator.SyntheticDatasetGenerator import SyntheticDatasetGenerator

class TimeSeriesDatasetGenerator(SyntheticDatasetGenerator):
	"""Concrete class used to generate time series.
	
	This class is used to create time series generator. It supports the
	generation of univariate time series with noise in input, trend, seasonality
	and point anomalies.
	"""
	
	def __init__(self, supervised : bool,
				 labels : list[str]):
		super().__init__(supervised, labels)
	
	def generate(self, num_points : int,
				 dimensions : int,
				 noise : list[str],
				 custom_noise : list[Callable[[], float]] = None,
				 verbose : bool = True,
				 sample_freq_seconds : float = 1.0,
				 trend : bool = False,
				 seasonality : bool = False,
				 trend_func : object = None,
				 seasonality_func : object = None) -> None:
		"""Generate a dataset with the given parameters.

		Arguments
		---------

		* num_points: an integer number representing the dimensionality of the
		dataset.
		* dimensions: number of dimensions of the dataset. CURRENTLY ONLY
		SUPPORTS 1.
		* noise: list of noise types for each dimension. Possible noise types
		are "Gaussian" and "custom".
		"""
		x = 1
		print("Done")