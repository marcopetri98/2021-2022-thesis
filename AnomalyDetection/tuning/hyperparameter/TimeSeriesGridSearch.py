from typing import Callable

import numpy as np
from skopt.space import Categorical, Integer

from utils.printing import print_step, print_warning
from tuning.hyperparameter.HyperparameterSearch import HyperparameterSearch
from tuning.hyperparameter.HyperparameterSearchSaver import \
	HyperparameterSearchSaver


class TimeSeriesGridSearch(HyperparameterSearch):
	"""Gird search over for time series datasets and models.
	
	Parameters
	----------
	load_checkpoint : bool, default=False
		If `true` it loads the checkpoint and continues the search.
	"""
	
	def __init__(self, parameter_space: list[Categorical | Integer],
				 model_folder_path: str,
				 search_filename: str,
				 cross_val_generator: object = None,
				 load_checkpoint: bool = False):
		super().__init__(parameter_space=parameter_space,
						 model_folder_path=model_folder_path,
						 search_filename=search_filename,
						 cross_val_generator=cross_val_generator)
		
		self.load_checkpoint = load_checkpoint
		self.tried_configs = {}
	
	def _run_optimization(self, objective_function: Callable[[np.ndarray,
															  np.ndarray,
															  np.ndarray,
															  np.ndarray,
															  dict], float],
						  verbose: bool = False):
		file_path = self.model_folder_path + self.search_filename
		config_saver = HyperparameterSearchSaver(file_path)
		
		if self.load_checkpoint:
			if verbose:
				print_step("Loading previous history of searches")
			
			history = config_saver.load_history()
			
			if verbose:
				print_step("Previous history has been loaded")
			
			self._run_grid_search(objective_function,
								  history=history,
								  callbacks=[config_saver],
								  verbose=verbose)
		else:
			if config_saver.history_exists():
				print_warning("There exists a checkpoint file!")
				print("Do you really want to overwrite it (you will lose it) [y/n]: ", end="")
				response = input()
				if response.lower() == "n" or response.lower() == "no":
					raise StopIteration("Stop")
			
			self._run_grid_search(objective_function,
								  callbacks=[config_saver],
								  verbose=verbose)

	def _run_grid_search(self, objective_function: Callable[[np.ndarray,
															 np.ndarray,
															 np.ndarray,
															 np.ndarray,
															 dict], float],
						 history: dict = None,
						 callbacks: list[Callable] = None,
						 verbose: bool = False) -> None:
		"""Runs grid search over the parameter space.
		
		Parameters
		----------
		history : ndarray, default=None
			It is the history of the search up to now, it is used in case an old
			search is resumed. The history has the following format: dict with
			as keys the name of the parameters and as values a list of length
			equal to the number of tries. If a parameter is not used in training,
			its value is None. Therefore, each column represent a configuration.
		
		callbacks : list[Callable], default=None
			It is a list of functions to call after a configuration has been
			tried. These call

		Returns
		-------
		None
		"""
		if verbose:
			print_step("Grid search has started")
		
		if history is not None:
			self._search_history = history.copy()
		
		# Input validation
		for parameter in self.parameter_space:
			if not (isinstance(parameter, Categorical) or
					isinstance(parameter, Integer)):
				raise ValueError("Cannot run grid search out of discrete values")
			
		# Build dict of param:possible values
		space = {}
		for parameter in self.parameter_space:
			if isinstance(parameter, Categorical):
				values = [category for category in parameter.categories]
			else:
				values = range(parameter.low, parameter.high + 1, 1)
				
			space[parameter.name] = values
			
		# Iterate over all possible configuration and call the objective
		sel_values = np.zeros(len(self.parameter_space), dtype=np.intc)
		has_finished = False
		while not has_finished:
			config = [space[self.parameter_space[idx].name][sel_values[idx]]
					  for idx in range(sel_values.shape[0])]
			
			# If the configuration has not been tried yet, objective is called
			if not self._find_history_entry(*config):
				score = self._objective_call(objective_function,
											 verbose,
											 *config)
				self._add_search_entry(score, *config)
				
				if callbacks is not None:
					for callback in callbacks:
						callback(search_history=self._search_history)
						
			# Change the configuration to try
			has_changed = False
			while not has_changed:
				for i in reversed(range(sel_values.shape[0])):
					sel_values[i] += 1
					if sel_values[i] == len(space[self.parameter_space[i].name]) and i != 0:
						sel_values[i] = 0
					else:
						has_changed = True
			
			# If all configuration has been tried, finish the loop
			if sel_values[0] == len(space[self.parameter_space[0].name]):
				has_finished = True
				
		if verbose:
			print_step("Grid search has tried all configurations")
