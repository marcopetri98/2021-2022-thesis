import abc
from abc import ABC
from typing import Callable

import numpy as np

from tuning.hyperparameter.IHyperparameterSearchResults import IHyperparameterSearchResults


class IHyperparameterSearch(ABC):
	"""Interface describing hyperparameter searchers.
    """
	
	@abc.abstractmethod
	def search(self, x,
			   y,
			   objective_function: Callable[[np.ndarray,
											 np.ndarray,
											 dict], float],
			   *args,
			   **kwargs) -> IHyperparameterSearchResults:
		"""Search the best hyperparameter values.
		
		Results of the search must be saved to file for persistence. The format
		the results must have is to be an array of shape (tries + 1, params + 1).
		Basically, the search contains a header of strings containing the names
		of the parameters searched and the name of the performance measure used
		to assess the model's quality. The other rows are composed of the values
		of the model's parameters and its performance with those parameters on
		the given dataset. Moreover, the performance of the model must always
		be in the first column.
		
		Parameters
		----------
		x : array-like of shape (n_samples, n_features)
			Array-like containing data over which to perform the search of the
			hyperparameters.
		
		y : array-like of shape (n_samples, n_target)
			Array-like containing targets of the data on which to perform the
			search of the hyperparameters.
		
		objective_function : Callable
			It is a function training the model and evaluating its performances.
			The first argument is the training set, the second argument is the
			validation set and the third argument is a dictionary of all the
			parameters of the model. The parameters must be used to instantiate
			the model.
		
		args
			Not used, present to allow multiple inheritance and signature change.
			
		kwargs
			Not used, present to allow multiple inheritance and signature change.

		Returns
		-------

		"""
		pass
	
	@abc.abstractmethod
	def print_search(self, *args,
					 **kwargs) -> None:
		"""Prints the results of the search stored on the file path.
		
		It uses the format specified by the fit function to read and print the
		results of the search at the specified file path. If there is no search
		file, an error will be raised.
		
		Parameters
		----------
		args
			Not used, present to allow multiple inheritance and signature change.
			
		kwargs
			Not used, present to allow multiple inheritance and signature change.

		Returns
		-------
		None
		"""
		pass
