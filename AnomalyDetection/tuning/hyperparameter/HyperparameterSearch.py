import abc
import time
from abc import ABC
from typing import Callable

import numpy as np
from sklearn.utils import check_X_y

from print_utils.printing import print_header, print_step
from tuning.hyperparameter.HyperparameterSearchResults import \
	HyperparameterSearchResults
from tuning.hyperparameter.IHyperparameterSearch import IHyperparameterSearch
from tuning.hyperparameter.IHyperparameterSearchResults import \
	IHyperparameterSearchResults


class HyperparameterSearch(IHyperparameterSearch, ABC):
	"""Abstract class implementing some hyperparameter search methods.
	
	Parameters
	----------
	parameter_space : list
		It is the space of the parameters to explore to perform hyperparameter
		search.

	model_folder_path : str
		It is the folder where the search results must be saved.

	search_filename : str
		It is the filename of the file where to store the results of the search.
		
	train_test_splitter : Callable
		It is the object performing the train-test split of the dataset.
	
	verbose : bool, default=True
			States if default printing or detailed printing must be performed.
	"""
	
	def __init__(self, parameter_space: list,
				 model_folder_path: str,
				 search_filename: str,
				 train_test_splitter: Callable[[np.ndarray], np.ndarray],
				 verbose: bool = True):
		super().__init__()
		
		self.parameter_space = parameter_space
		self.model_folder_path = model_folder_path
		self.search_filename = search_filename
		self.train_test_splitter = train_test_splitter
		self.verbose = verbose
		
		self._data = None
		self._data_labels = None
		self._search_history = None
	
	def search(self, x,
			   y,
			   objective_function: Callable[[np.ndarray,
											 np.ndarray,
											 dict], float],
			   *args,
			   **kwargs) -> IHyperparameterSearchResults:
		check_X_y(x, y)
		x = np.array(x)
		y = np.array(y)
		
		start_time = time.time()
		self._data, self._data_labels = x, y
		
		if self.verbose:
			print_header("Starting the hyperparameter search")
		
		self._run_optimization(objective_function)
		
		final_history = [self._create_first_row(), *self._search_history]
		final_history = np.array(final_history, dtype=object)
		np.save(self.model_folder_path + self.search_filename, final_history)
		
		results = HyperparameterSearchResults(np.array(final_history))
		
		self._data, self._data_labels, self._search_history = None, None, None
		if self.verbose:
			print_step("Search lasted for: " + str(time.time() - start_time))
			print_header("Hyperparameter search ended")
			
		return results
	
	def print_search(self, *args, **kwargs) -> None:
		tries = np.load(self.model_folder_path + self.search_filename + ".npy",
						allow_pickle=True)
		indices = (np.argsort(tries[1:, 0]) + 1).tolist()
		indices.reverse()
		indices.insert(0, 0)
		tries = tries[np.array(indices)]

		print("Total number of tries: ", tries.shape[0] - 1)
		first = True
		for config in tries:
			if first:
				first = False
			else:
				text = ""
				for i in range(len(config)):
					text += str(tries[0, i])
					text += ": " + str(config[i]) + " "
				print(text)
	
	@abc.abstractmethod
	def _run_optimization(self, objective_function: Callable[[np.ndarray,
															  np.ndarray,
															  dict], float]):
		"""Runs the optimization search.
		
		Parameters
		----------
		objective_function : Callable
			The objective function to minimize.
			
		Notes
		-----
		See :meth:`~tuning.hyperparameter.HyperparameterSearch.search` for more
		details about objective_function.
		"""
		pass
	
	@abc.abstractmethod
	def _objective_call(self, objective_function: Callable[[np.ndarray,
															np.ndarray,
															dict], float],
						*args) -> float:
		"""The function wrapping the loss to minimize.
		
		The function wraps the loss to minimize passed to the object by
		manipulating the dataset to obtain training and validation sets and by
		saving results to the search history.

		Parameters
		----------
		args : list
			The list of all the parameters passed to the function to be able
			to run the algorithm.
			
		objective_function : Callable
			The objective function to minimize.

		Returns
		-------
		function_value: float
			The value of the computed objective function.
			
		Notes
		-----
		See :meth:`~tuning.hyperparameter.HyperparameterSearch.search` for more
		details about objective_function.
		"""
		pass
	
	def _add_search_entry(self, score, args) -> None:
		"""It adds an entry to the search history.
		
		Parameters
		----------
		score : float
			The score of this configuration.
		
		args
			The passed arguments to the optimization function.

		Returns
		-------
		None
		"""
		vals = [score]
		for arg in args:
			vals.append(arg)
		
		if self._search_history is None:
			self._search_history = [vals]
		else:
			self._search_history.append(vals)
	
	def _build_input_dict(self, args) -> dict:
		"""Build the dictionary of the parameters.

		Parameters
		----------
		args
			The passed arguments to the optimization function.

		Returns
		-------
		parameters : dict[str]
			Dictionary with parameter names as keys and their values as values.
		"""
		params = {}
		for i in range(len(args)):
			params[self.parameter_space[i].name] = args[i]
		return params
	
	def _create_first_row(self) -> list:
		"""Builds the list of the parameters for the training process.

		Returns
		-------
		parameters : list
			The list of all the parameters used for the training process.

		Returns
		-------
		None
		"""
		parameters = ["Performance"]
		for parameter in self.parameter_space:
			parameters.append(parameter.name)
		return parameters
