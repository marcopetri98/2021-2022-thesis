from models.Learner import Learner


class UnsupervisedLearner(Learner):
	"""Abstract class used to define an unsupervised learner"""
	
	def __init__(self):
		super().__init__()
