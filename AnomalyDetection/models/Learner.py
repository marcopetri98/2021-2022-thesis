import numpy as np


class Learner(object):
	"""Abstract class used to define a learner"""
	
	def __init__(self):
		super().__init__()
		
	def fit(self, data: np.ndarray,
			*args,
			**kwargs) -> None:
		pass
	
	def get_params(self) -> dict:
		return vars(self)
	
	def set_params(self, *args,
				   **kwargs) -> None:
		pass
