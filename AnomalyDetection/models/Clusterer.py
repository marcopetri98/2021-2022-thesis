# Python imports
import abc

# External imports

# Project imports
from models.UnsupervisedLearner import UnsupervisedLearner


class Clusterer(UnsupervisedLearner):
	"""Abstract class used to define an unsupervised learner"""
	
	def __init__(self):
		super().__init__()
	
	@abc.abstractmethod
	def get_centroids(self, *args,
					  **kwargs) -> None:
		pass
	
	@abc.abstractmethod
	def get_clustering(self, *args,
					 **kwargs) -> None:
		pass
