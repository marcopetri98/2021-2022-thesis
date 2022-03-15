# Python imports
import os

# External imports
import json
import numpy as np


class Learner(object):
	"""Abstract class used to define a learner"""
	
	def __init__(self):
		super().__init__()
		models_dir = os.path.dirname(__file__)
		file_path = os.path.join(models_dir, 'errors.json')
		errors_file = open(file_path)
		self.errors = json.load(errors_file)
		errors_file.close()
		
	def fit(self, data: np.ndarray,
			*args,
			**kwargs) -> None:
		pass
	
	def get_params(self) -> dict:
		return vars(self)
	
	def set_params(self, *args,
				   **kwargs) -> None:
		pass
