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
	
	def __init__(self, history: list):
		super().__init__()
		
		self.history = history
	
	def get_best_score(self) -> float:
		return np.min(self.history[1:, 0])
	
	def get_num_iterations(self) -> int:
		return len(self.history) - 1
	
	def get_history(self) -> list:
		return self.history.copy()
	
	def print_search(self, *args, **kwargs) -> None:
		if len(self.history) == 0:
			print("There is no search")
		else:
			scores = [x[0] for x in self.history[1::]]
			tries = np.array(scores)
			indices = (np.argsort(tries) + 1).tolist()
			indices.reverse()
			indices.insert(0, 0)
			tries = [self.history[i] for i in indices]
	
			print("Total number of tries: ", len(tries) - 1)
			first = True
			for config in tries:
				if first:
					first = False
				else:
					text = ""
					for i in range(len(config)):
						text += str(tries[0][i])
						text += ": " + str(config[i]) + " "
					print(text)
