import abc
from abc import ABC

import numpy as np


class IHyperparameterSearchResults(ABC):
	"""Interface to represent hyperparameter search result exposed methods.
    """
	
	@abc.abstractmethod
	def get_best_score(self) -> float:
		"""Finds the best score obtained on the search and returns it.
		
		Returns
		-------
		best_score: float
			The best score obtained on the search.
		"""
		pass
	
	@abc.abstractmethod
	def get_num_iterations(self) -> int:
		"""Returns the number of iterations passed searching.
		
		Returns
		-------
		num_iter : int
			The number of iterations performed on search.
		"""
		pass
	
	@abc.abstractmethod
	def get_history(self) -> np.ndarray:
		"""Returns the history of the hyperparameter search.
		
		Returns
		-------
		history : ndarray of shape (n_iterations, 2)
			The history of the search in which each element is composed of two
			values: score and tried parameters as a dictionary.
		"""
		pass
