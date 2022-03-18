# Python imports
from abc import ABC

# External imports

# Project imports
from models.Learner import Learner


class UnsupervisedLearner(Learner, ABC):
	"""Abstract class used to define an unsupervised learner"""
	def __init__(self):
		super().__init__()
