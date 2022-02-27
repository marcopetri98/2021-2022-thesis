from math import sin, pi, cos
from typing import Callable


class Seasonality(object):
	"""Class defining the seasonality component for a time series.

	Members
	-------

	* type: string equal to one of "custom", "sine", "cosine",
	"positive_triangular", "negative_triangular", "triangular". It defines the
	type os seasonal component.
	* amplitude: amplitude of the seasonal component. E.g., if the amplitude is 
	10 a sine seasonality will vary from -5 to 5.
	* seasonality_length: length in seconds of the seasonal components
	* sample_freq: float representing the number of samples per second in the 
	time series. E.g., if we have 1 sample each 5 seconds we have 1/5 of 
	frequency, i.e., frequency = 0.2.
	* seasonality_func [optional]: if the function representing the seasonality
	in case the type os the seasonality is custom.

	Notes
	-----
	
	It is important to remember that the seasonality functions must be periodic
	and to them it will be passed an input varying from 0 to 1 representing how
	much they're close to the start or the end of the period.
	"""
	ALLOWED_SEASONALITY = ["custom",
							"sine",
							"cosine",
							"positive_triangular",
							"negative_triangular",
							"triangular"]
	
	def __init__(self, type : str,
				 amplitude : float,
				 seasonality_length : float,
				 sample_freq_seconds : float,
				 seasonality_func : Callable[[float], None] = None):
		if type not in Seasonality.ALLOWED_SEASONALITY:
			raise ValueError("The seasonality type must be one of ",
								 Seasonality.ALLOWED_SEASONALITY)
		
		super().__init__()
		self.type = type
		self.amplitude = amplitude
		self.length = seasonality_length
		self.sample_freq = sample_freq_seconds
		self.seasonality_func = seasonality_func

	def compute_seasonal_value(self, elapsed_seconds : float) -> float:
		"""Computes the value of the trend component of the time series.

		The seasonal component is computed by computing the position in the
		period of the seasonal function. Then, the seasonal function (i.e.
		periodic function) is called giving to it the position on its period.
		Therefore, it is possible to perfectly determine the seasonal component
		of the time series.

		Parameters
		----------

		* elapsed_seconds: float representing the number of elapsed seconds from
		the start of the time series.

		Returns
		-------

		* float: the absolute value of the seasonal component of the time series.
		"""
		if elapsed_seconds < 0:
			raise ValueError("The elapsed time in seconds must be greater"
			" or equal 0")

		function_input = (elapsed_seconds % self.length) / self.length
		
		if self.type == "custom":
			seasonality = self.seasonality_func(function_input)
		else:
			seasonality = self.__compute_seasonality(elapsed_seconds)

		return seasonality
			
	def __compute_seasonality(self, period_position : float) -> float:
		"""Performs the computation of the seasonality.
		
		Parameters
		----------

		* period_position: float representing the position on the period of the
		function.

		Returns
		-------

		* float: the absolute seasonal value.
		"""
		if period_position < 0 or period_position > 1:
			raise ValueError("The attribute period_position must be between"
			" 0 and 1 as it represents percentage")
		seasonality_value = 0
		
		match self.type:
			case "sine":
				seasonality_value = (self.amplitude / 2) \
									* sin(period_position * 2 * pi)
			
			case "cosine":
				seasonality_value = (self.amplitude / 2) \
									* cos(period_position * 2 * pi)
			
			case "positive_triangular":
				x = period_position * self.length
				if x <= self.length/2:
					seasonality_value = (2 * self.amplitude * x) \
										/ self.length
				else:
					seasonality_value = 2 * self.amplitude \
										- (2 * self.amplitude * x) \
										/ self.length
			
			case "negative_triangular":
				x = period_position * self.length
				if x <= self.length/2:
					seasonality_value = - (2 * self.amplitude * x) \
										/ self.length
				else:
					seasonality_value = - 2 * self.amplitude \
										+ (2 * self.amplitude * x) \
										/ self.length
			
			case "triangular":
				x = period_position * self.length
				if x <= self.length/2:
					seasonality_value = self.amplitude / 2 \
										- (2 * self.amplitude * x) \
										/ self.length
				else:
					seasonality_value = - 3 * self.amplitude / 2 \
										+ (2 * self.amplitude * x) \
										/ self.length
			
		return seasonality_value