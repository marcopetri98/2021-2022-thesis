import numpy as np

from models.Learner import Learner


class SupervisedLearner(Learner):
	"""Abstract class used to define a supervised learner"""
	
	def __init__(self):
		super().__init__()
	
	def fit(self, data: np.ndarray,
			labels: np.ndarray = None,
			*args,
			**kwargs) -> None:
		pass
	
	def set_params(self, *args,
				   **kwargs) -> None:
		pass