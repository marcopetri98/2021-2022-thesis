# Python imports
import abc

# External imports
import numpy as np

# Project imports
from base.BaseObject import BaseObject


class Predictor(BaseObject):
	"""Abstract class used to define a learner"""
	
	def __init__(self):
		super().__init__()
	
	@abc.abstractmethod
	def predict(self, test: np.ndarray,
			*args,
			**kwargs) -> None:
		pass