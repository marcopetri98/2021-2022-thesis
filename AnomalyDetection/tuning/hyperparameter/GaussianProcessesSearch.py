from typing import Callable

import numpy as np
import skopt
from scipy.optimize import OptimizeResult
from skopt.callbacks import CheckpointSaver
from skopt.space import Categorical, Integer

from tuning.hyperparameter.HyperparameterSearch import HyperparameterSearch
from tuning.hyperparameter.IHyperparameterSearchResults import \
	IHyperparameterSearchResults


class GaussianProcessesSearch(HyperparameterSearch):
	"""HyperparameterSearch is a class used to search the hyperparameters.
	
	Parameters
	----------
	load_checkpoint : bool, default=False
		If `true` it loads the checkpoint and continues the search.
	"""
	__INVALID_GP_KWARGS = ["func", "dimensions", "x0", "y0"]

	def __init__(self, parameter_space: list[Categorical | Integer],
				 model_folder_path: str,
				 search_filename: str,
				 cross_val_generator: object = None,
				 load_checkpoint: bool = False,
				 gp_kwargs: dict = None):
		super().__init__(parameter_space=parameter_space,
						 model_folder_path=model_folder_path,
						 search_filename=search_filename,
						 cross_val_generator=cross_val_generator)
		
		if gp_kwargs is not None and len(set(self.__INVALID_GP_KWARGS).difference(gp_kwargs.keys())) != len(self.__INVALID_GP_KWARGS):
			raise ValueError("The following keywords of gp_minimize cannot be "
							 "passed since are automatically managed by the "
							 "class: {}".format(self.__INVALID_GP_KWARGS))

		self.test = 0
		self.load_checkpoint = load_checkpoint
		self.gp_kwargs = dict(gp_kwargs) if gp_kwargs is not None else {}
		
		self.__minimized_objective = None
		self.__run_opt_verbose = False
		
	def search(self, x,
			   y,
			   objective_function: Callable[[np.ndarray,
											 np.ndarray,
											 np.ndarray,
											 np.ndarray,
											 dict], float],
			   verbose: bool = False,
			   *args,
			   **kwargs) -> IHyperparameterSearchResults:
		"""
		Parameters
		----------
		objective_function : Callable
			The objective function of the hyperparameter search to be minimized.
		"""
		return super().search(x, y, objective_function, verbose, *args, **kwargs)

	def _run_optimization(self, objective_function: Callable[[np.ndarray,
															  np.ndarray,
															  np.ndarray,
															  np.ndarray,
															  dict], float],
						  verbose: bool = False) -> OptimizeResult:
		self.__minimized_objective = objective_function
		self.__run_opt_verbose = verbose
		
		file_path = self.model_folder_path + self.search_filename
		checkpoint_saver = CheckpointSaver(file_path + ".pkl", compress=9)

		callbacks = [checkpoint_saver]
		if "callback" in self.gp_kwargs.keys():
			callbacks.append(self.gp_kwargs["callback"])
			del self.gp_kwargs["callback"]

		if self.load_checkpoint:
			previous_checkpoint = skopt.load(file_path + ".pkl")
			x0 = previous_checkpoint.x_iters
			y0 = previous_checkpoint.func_vals
			
			if x0[0].shape[0] != len(self.parameter_space):
				raise ValueError("If you are continuing a search, do not change"
								 " the parameter space.")
			
			for config, retval in zip(x0, y0):
				self._search_history = None
				self._add_search_entry(retval, *config)
			
			results = skopt.gp_minimize(self._gaussian_objective,
										self.parameter_space,
										x0=x0,
										y0=y0,
										callback=callbacks,
										**self.gp_kwargs)
		else:
			results = skopt.gp_minimize(self._gaussian_objective,
										self.parameter_space,
										callback=callbacks,
										**self.gp_kwargs)

		self.__run_opt_verbose = False
		self.__minimized_objective = None

		return results

	def _gaussian_objective(self, args):
		"""Respond to a call with the parameters chosen by the Gaussian Process.
		
		Parameters
		----------
		args : list
			The space parameters chosen by the gaussian process.

		Returns
		-------
		configuration_score : float
			The score of the configuration to be minimized.
		"""
		score = self._objective_call(self.__minimized_objective,
									 self.__run_opt_verbose,
									 *args)
		self._add_search_entry(score, *args)
		return score
