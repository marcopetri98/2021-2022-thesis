import numpy as np

from tuning.hyperparameter.IHyperparameterSearchResults import IHyperparameterSearchResults


class HyperparameterSearchResults(IHyperparameterSearchResults):
	"""It stores the results of a hyperparameter search.
	
	Parameters
	----------
	history : ndarray of shape (n_search + 1, params + 1)
		It represents the history of the hyperparameter search. The first row is
		the name of the parameters represented by the columns of the history.
	"""
	
	def __init__(self, history: np.ndarray):
		super().__init__()
		
		self.history = history
	
	def get_best_score(self) -> float:
		return np.min(self.history[1:, 0])
	
	def get_num_iterations(self) -> int:
		return self.history.shape[0] - 1
	
	def get_history(self) -> np.ndarray:
		return self.history
