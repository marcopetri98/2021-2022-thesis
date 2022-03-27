# Python imports
import colorama
from colorama import Fore, Style

# External imports

# Project imports


def print_header(heading: str, separator: str = "=") -> None:
	"""Prints an heading.

	Parameters
	----------
	heading: str
		The text to be printed as heading.

	separator: str, default="="
		The separator to be used to limit the heading.

	Returns
	-------
	None
	"""
	# Checks assumption
	if len(separator) != 1:
		raise ValueError("The separator must be a single character")

	string_to_print = "=" * (len(heading) + 4)
	string_to_print = string_to_print + "\n"
	string_to_print = string_to_print + "= " + heading + " =\n"
	string_to_print = string_to_print + "=" * (len(heading) + 4)

	print(string_to_print)


def print_step(text: str) -> None:
	"""Prints a step of execution.

	Parameters
	----------
	text: str
		A string representing the execution step.

	Returns
	-------
	None
	"""
	if len(text) == 0:
		raise ValueError("An execution step print must have a text")

	step_print = "Execution: "
	step_print += text

	print(step_print)


def print_warning(warning_text: str) -> None:
	"""Prints a warning on screen.

	Parameters
	----------
	warning_text: str
		A string representing the execution step.

	Returns
	-------
	None
	"""
	if len(warning_text) == 0:
		raise ValueError("The warning must have a text")

	colorama.init()
	warning = "WARNING: "
	warning += warning_text

	print(Fore.RED + Style.BRIGHT + warning + Style.RESET_ALL)
