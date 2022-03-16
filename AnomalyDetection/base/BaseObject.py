# Python imports
from abc import ABC
import os

# External imports
import json

# Project imports

def mix_keys(class1, class2) -> list[str]:
	"""Mix the error keys of two classes maintaining order"""
	keys = []
	class1_elems = len(class1.ERROR_KEY)
	class2_elems = len(class2.ERROR_KEY)
	if class1_elems > class2_elems:
		for i in range(class1_elems):
			keys.append(class1.ERROR_KEY[i])
			if i < class2_elems and class1.ERROR_KEY[i] != class2.ERROR_KEY[i]:
				keys.append(class2.ERROR_KEY[i])
	elif class1_elems < class2_elems:
		for i in range(class2_elems):
			if i < class1_elems:
				keys.append(class1.ERROR_KEY[i])
			if class1.ERROR_KEY[i] != class2.ERROR_KEY[i]:
				keys.append(class2.ERROR_KEY[i])
	else:
		for i in range(class1_elems):
			keys.append(class1.ERROR_KEY[i])
			if class1.ERROR_KEY[i] != class2.ERROR_KEY[i]:
				keys.append(class2.ERROR_KEY[i])
	return keys


class BaseObject(ABC):
	"""Base object for the project objects."""
	ERROR_KEY = ["base"]
	
	def __init__(self):
		super().__init__()
		models_dir = os.path.dirname(__file__)
		file_path = os.path.join(models_dir, 'errors.json')
		errors_file = open(file_path)
		self.errors = json.load(errors_file)
		errors_file.close()
	
	def get_params(self) -> dict:
		"""Get the parameters of the model."""
		return vars(self)
	
	def set_params(self, *args,
				   **kwargs) -> None:
		"""Sets the parameters of the model.

		Notes
		-----
		Parameters have the same names as the ones in the object creation.
		"""
		self._check_assumptions(kwargs)
		for key in kwargs.keys():
			# Check if the attribute exists
			getattr(self, key)
			setattr(self, key, kwargs[key])
	
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
		for i in range(len(self.ERROR_KEY)):
			try:
				curr = - (i + 1)
				return self.errors[self.ERROR_KEY[curr]][error_name]
			except KeyError:
				pass
		
		return self.errors["base"]["generic"]
