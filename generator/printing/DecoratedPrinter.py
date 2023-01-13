

class DecoratedPrinter(object):
	"""Class used to define decorated printings for verbose prints.
	"""

	@staticmethod
	def print_heading(text : str, separator : str = "=") -> None:
		"""Prints an heading surrounded by separator.

		Parameters
		----------

		* text: the heading text to print.
		* separator: a character used to print the borders.
		"""
		# Checks assumption
		if len(separator) != 1:
			raise ValueError("The separator must be a single character")

		string_to_print = "=" * (len(text) + 4)
		string_to_print = string_to_print + "\n"
		string_to_print = string_to_print + "= " + text + " =\n"
		string_to_print = string_to_print + "=" * (len(text) + 4)
		
		print(string_to_print)

	@staticmethod
	def print_step(text : str) -> None:
		"""Prints a step of execution.

		Parameters
		----------

		* text: a string representing the execution step.
		"""
		if len(text) == 0:
			raise ValueError("An execution step print must have a text")

		step_print = "Execution: "
		step_print += text

		print(step_print)