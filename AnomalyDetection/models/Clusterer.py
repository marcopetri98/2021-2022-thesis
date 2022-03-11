import numpy as np

from models.UnsupervisedLearner import UnsupervisedLearner


class Clusterer(UnsupervisedLearner):
	"""Abstract class used to define an unsupervised learner"""
	
	def __init__(self):
		super().__init__()
	
	def fit(self, data: np.ndarray,
			*args,
			**kwargs) -> None:
		pass
	
	def get_clusters(self, *args,
					 **kwargs) -> None:
		pass
	
	def get_clustering(self, *args,
					 **kwargs) -> None:
		pass
	
	def set_params(self, *args,
				   **kwargs) -> None:
		pass
