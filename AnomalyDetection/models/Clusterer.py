from models.UnsupervisedLearner import UnsupervisedLearner


class Clusterer(UnsupervisedLearner):
	"""Abstract class used to define an unsupervised learner"""
	
	def __init__(self):
		super().__init__()
	
	def get_centroids(self, *args,
					  **kwargs) -> None:
		pass
	
	def get_clustering(self, *args,
					 **kwargs) -> None:
		pass
