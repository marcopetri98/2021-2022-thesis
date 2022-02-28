from typing import Callable


class Trend(object):
	"""Class defining the trend component for a time series.

	Members
	-------

	* type: a string representing the type of trend for the time series. It must
	be on of "custom", "linear", "quadratic", "cubic", "monomial", "exponential".
	* starting_point: a float value representing the starting point for the time
	 series in terms of trend value (it is not the first sample value).
	* arguments: it is a list of the arguments needed by the standard trend
	function definition. None if type == "custom".
	* custom_func [optional]: it is the custom function used to define the trend
	 for the time series.

	Examples
	--------
	The following example defines a linear trend for a time series with 
	coefficient 1. It means that the trend value is f(x) = x and 5 is the 
	starting point.

	::

		trend = Trend("linear", 5, [1])
		print("Trend value after 10 seconds", trend.compute_trend_value(10))

	The following example defines a monomial term of degree 5 with coefficient 
	2. It means that the tren value is f(x) = 2 x^5 and 4 is the starting point.

	::

		trend = Trend("monomial", 4, [2, 5])
		print("Trend value after 10 seconds", trend.compute_trend_value(10))

	The following example defines an exponential term of with base 2 with 
	coefficient 1. It means that the tren value is f(x) = 2^x and 4 is the
	starting point.

	::

		trend = Trend("exponential", 4, [1, 2])
		print("Trend value after 10 seconds", trend.compute_trend_value(10))

	Notes
	-----

	The custom trend component can be any function you define. It must take as
	input the number of elapsed seconds from the start of the time series and it
	 must output the trend value for that time.
	"""
	ALLOWED_TRENDS = ["custom",
					  "linear",
					  "quadratic",
					  "cubic",
					  "monomial",
					  "exponential"]
	
	def __init__(self, trend_type: str,
				 starting_point: float,
				 function_args: list[float] = None,
				 custom_func: Callable[[float], None] = None):
		if trend_type not in Trend.ALLOWED_TRENDS:
			raise ValueError("The trend type must be one of ",
								 Trend.ALLOWED_TRENDS)

		super().__init__()
		self.type = trend_type
		self.starting_point = starting_point
		self.arguments = function_args
		self.custom_func = custom_func

	def compute_trend_value(self, elapsed_seconds: float) -> float:
		"""Compute the value of the trend component of the time series.

		The method computes the trend component for the time series using the 
		chosen function and by considering the number of seconds passed from the
		 start of the time series.
		
		Parameters
		----------

		* elapsed_seconds: float representing the number of elapsed seconds from
		 the start of the time series.

		Returns
		-------

		* float: the absolute value of the trend component of the time series.
		"""
		if elapsed_seconds < 0:
			raise ValueError("The elapsed time in seconds must be greater"
			" or equal 0")

		trend_value = 0

		if self.type == "custom":
			trend_value += self.custom_func(elapsed_seconds)
		else:
			trend_value += self.__compute_trend(elapsed_seconds)

		return trend_value

	def __compute_trend(self, elapsed_seconds: float) -> float:
		"""Performs the computation of the trend component.

		Parameters
		----------

		* elapsed_seconds: float representing the number of elapsed seconds from
		 the start of the time series.

		Returns
		-------

		* float: the absolute value of the trend component of the time series.
		"""
		if elapsed_seconds < 0:
			raise ValueError("The elapsed time in seconds must be greater"
			" or equal 0")
			
		trend_value = 0

		match self.type:
			case "linear":
				coefficient = self.arguments[0]
				trend_value = coefficient * elapsed_seconds

			case "quadratic":
				coefficient = self.arguments[0]
				trend_value = coefficient * (elapsed_seconds ** 2)

			case "cubic":
				coefficient = self.arguments[0]
				trend_value = coefficient * (elapsed_seconds ** 3)

			case "monomial":
				coefficient = self.arguments[0]
				power = self.arguments[1]
				trend_value = coefficient * (elapsed_seconds ** power)

			case "exponential":
				coefficient = self.arguments[0]
				base = self.arguments[1]
				trend_value = coefficient * (base ** elapsed_seconds)

		return trend_value
