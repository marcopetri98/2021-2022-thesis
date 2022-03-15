import numpy as np

from models.SupervisedLearner import SupervisedLearner
from models.UnsupervisedLearner import UnsupervisedLearner


class AnomalyLearner(UnsupervisedLearner, SupervisedLearner):
	"""Abstract class used to define an anomaly learner"""
	
	def __init__(self):
		super().__init__()
	
	def get_anomaly_scores(self, *args,
						   **kwargs) -> np.ndarray:
		pass
	
	def get_anomalies(self, *args,
					  **kwargs) -> np.ndarray:
		pass