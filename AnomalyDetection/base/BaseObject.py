# Python imports
from abc import ABC
import os

# External imports
import json

# Project imports


class BaseObject(ABC):
	"""Base object for the project objects."""
	
	def __init__(self):
		super().__init__()
		models_dir = os.path.dirname(__file__)
		file_path = os.path.join(models_dir, 'errors.json')
		errors_file = open(file_path)
		self.errors = json.load(errors_file)
		errors_file.close()
	
	def _check_assumptions(self, *args,
						   **kwargs) -> None:
		"""Checks if the assumption about the specified variable are true."""
		return
	
	def _raise_error(self, error_name: str) -> str:
		"""Prints the error message given the error name key.

		Parameters
		----------
		error_name : str
			The key for the error type of that object.
		"""
		try:
			return self.errors[error_name]
		except KeyError:
			pass
		
		return self.errors["generic"]
