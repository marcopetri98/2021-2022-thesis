# python libraries
from statistics import mean
from typing import Union, Tuple, Callable
from dataclasses import dataclass
import inspect
import random

# external libraries
import numpy as np

# project libraries

@dataclass(frozen=True)
class AnomalyPrints(object):
	ERROR_IMPL_ANOMALY = "Anomaly type not implemented"
	ERROR_TOO_SHORT = "A collective anomaly must have at least 2 points"
	ERROR_PARAM_MATCH = "The given parameters does not match with the type"

class Anomaly(object):
	"""A class representing the implementation of an anomaly type

	Parameters
	----------

	* anomaly_type: a string representing the type of anomaly.
	* collective_type: type of collective anomaly in case it is collective.
	* length: an int representing the length of the anomaly.
	* params: the list of the parameters of the anomaly type.

	Notes
	-----

	Point anomalies take as parameters a series of offsets describing them.

	Collective anomalies take as parameters the offset/constant values (same as
	for a point anomalies, i.e., a list of the possible offsets or constants) or
	a callable function taking the number of points to generate as input (int)
	and producing as output a list of values to be added to the time series. The
	mean takes as input the number of previous points to consider computing the
	mean at which the collective anomaly will be constant.
	"""

	ALLOWED_TYPES = ["point",
					 "collective"]
	ALLOWED_COLLECTIVE = ["offset",
						  "mean",
						  "function",
						  "constant"]

	def __init__(self, anomaly_type: str,
				 collective_type: str = None,
				 anomaly_length: int = 1,
				 anomaly_params: list[Union[float, Callable[[int], list[float]]]] = None):
		if anomaly_type not in self.ALLOWED_TYPES:
			raise ValueError(AnomalyPrints.ERROR_IMPL_ANOMALY)
		elif collective_type is not None and collective_type not in self.ALLOWED_COLLECTIVE:
			raise ValueError(AnomalyPrints.ERROR_IMPL_ANOMALY)
		elif anomaly_type == "collective" and anomaly_length < 2:
			raise ValueError(AnomalyPrints.ERROR_TOO_SHORT)
		elif not Anomaly.__check_anomaly_params(anomaly_type,
												anomaly_params,
												collective_type):
			raise ValueError(AnomalyPrints.ERROR_PARAM_MATCH)

		super().__init__()
		self.anomaly_type = anomaly_type
		self.collective_type = collective_type
		self.length = anomaly_length
		self.params = anomaly_params

	def compute_anomaly(self, dataset: np.ndarray,
						position: Union[int, Tuple[int]]) -> Union[float, list[float]]:
		"""Compute the anomaly in the position

		Parameters
		----------

		* dataset: the dataset on which we need to add the anomaly.
		* position: the position on which we want to add the anomaly.
		"""
		is_univariate = True

		if isinstance(position, tuple):
			is_univariate = False

		match self.anomaly_type:
			case "point":
				return random.choice(self.params) + dataset[position]

			case "collective":
				match self.collective_type:
					case "offset":
						offset = random.choice(self.params)
						return dataset[position:position+self.length] + offset

					case "mean":
						avg = 0
						if is_univariate:
							avg += mean(dataset[position-self.params[0]:position])
						else:
							pos = list(position)
							start = pos
							start[0] -= self.params[0]
							avg += mean(dataset[start:pos])
						return [avg] * self.length

					case "constant":
						value = random.choice(self.params)
						return [value] * self.length

					case "function":
						return self.params[0](self.length)

	@staticmethod
	def __check_anomaly_params(anomaly_type: str,
							   anomaly_params: list[float],
							   collective_type: str) -> bool:
		"""Checks if the parameters are right for that anomaly type"""
		match anomaly_type:
			case "point":
				if len(anomaly_params) > 0 and 0 not in anomaly_params:
					return True

			case "collective":
				match collective_type:
					case "mean":
						if len(anomaly_params) == 1 and anomaly_params[0] > 0:
							return True

					case "offset":
						if len(anomaly_params) > 0 and 0 not in anomaly_params:
							return True

					case "function":
						if (len(anomaly_params) == 1 and
								callable(anomaly_params[0]) and
								not inspect.isclass(anomaly_params[0])):
							return True

					case "constant":
						if len(anomaly_params) > 0 and 0 not in anomaly_params:
							return True

		return False
