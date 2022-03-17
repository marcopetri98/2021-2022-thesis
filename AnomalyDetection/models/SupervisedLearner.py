# Python imports
import abc

# External imports
import numpy as np

# Project imports
from models.Learner import Learner


class SupervisedLearner(Learner):
	"""Abstract class used to define a supervised learner"""
	ERROR_KEY = Learner.ERROR_KEY.copy() + ["supervised"]
	
	def __init__(self):
		super().__init__()
	
	@abc.abstractmethod
	def fit(self, train: np.ndarray,
			labels: np.ndarray = None,
			*args,
			**kwargs) -> None:
		pass