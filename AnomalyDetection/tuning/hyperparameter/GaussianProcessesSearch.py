# Python imports
from typing import Union, Callable

import skopt
from scipy.optimize import OptimizeResult
from sklearn import metrics
from skopt.callbacks import CheckpointSaver
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from utils.printing import print_header, print_step
from tuning.hyperparameter.HyperparameterSearch import HyperparameterSearch


class GaussianProcessesSearch(HyperparameterSearch):
	"""HyperparameterSearch is a class used to search the hyperparameters.

	Parameters
	----------
	estimator: BaseEstimator
		The scikit-learn estimator on which to apply hyperparameter search.

	calls: int, default= 10

	initial_starts: int, default= 10

	cross_validation: bool or Callable, default=True

	train_and_test: bool default=True

	load_checkpoint: bool default=False
	"""

	def __init__(self, estimator: BaseEstimator,
				 parameter_space: list,
				 model_folder_path: str,
				 search_filename: str,
				 cross_val_generator: Callable,
				 verbose: bool = True,
				 calls: int = 10,
				 initial_starts: int = 10,
				 cross_validation: Union[bool, Callable] = True,
				 train_and_test: bool = True,
				 load_checkpoint: bool = False):
		super().__init__(parameter_space=parameter_space,
						 model_folder_path=model_folder_path,
						 search_filename=search_filename,
						 cross_val_generator=cross_val_generator,
						 verbose=verbose)

		self.estimator = estimator
		self.calls = calls
		self.initial_starts = initial_starts
		self.cross_validation = cross_validation
		self.train_and_test = train_and_test
		self.load_checkpoint = load_checkpoint

	def _run_optimization(self) -> OptimizeResult:
		"""Runs the optimization search and return the results.

		Returns
		-------
		search_results: OptimizeResult
			The results of the hyperparameter search.
		"""
		file_path = self.model_folder_path + self.search_filename
		checkpoint_saver = CheckpointSaver(file_path + ".pkl", compress=9)

		if self.load_checkpoint:
			previous_checkpoint = skopt.load(file_path + ".pkl")
			x0 = previous_checkpoint.x_iters
			y0 = previous_checkpoint.func_vals
			results = skopt.gp_minimize(self._objective_call,
										self.parameter_space,
										x0=x0,
										y0=y0,
										n_calls=self.calls,
										n_initial_points=self.initial_starts,
										callback=[checkpoint_saver])
		else:
			results = skopt.gp_minimize(self._objective_call,
										self.parameter_space,
										n_calls=self.calls,
										n_initial_points=self.initial_starts,
										callback=[checkpoint_saver])

		return results

	def _objective_call(self, *args) -> float:
		estimator: BaseEstimator = clone(self.estimator)
		params = self._build_input_dict(*args)
		estimator.set_params(**params)

		if self.verbose:
			print_step("running configuration with params: " + str(params))

		if isinstance(self.cross_validation, bool):
			if self.cross_validation:
				k_fold_score = 0
				cross_val_gen = KFold(n_splits=5)

				for train, test in cross_val_gen.split(self.data, self.data_labels):
					try:
						if self.train_and_test:
							estimator.fit(self.data[train], self.data_labels[train])
							y_pred = estimator.predict(self.data[test])
						else:
							y_pred = estimator.fit_predict(self.data[test])

						k_fold_score += metrics.f1_score(self.data_labels[test],
														 y_pred,
														 zero_division=0)
					except ValueError:
						pass
				k_fold_score = k_fold_score / cross_val_gen.get_n_splits()

				if self.verbose:
					print_step("run with score: " + str(k_fold_score))

				return 1 - k_fold_score
			else:
				test_dim = int(self.data.shape[0] * 0.2)

				score = 0
				try:
					if self.train_and_test:
						estimator.fit(self.data[0:-test_dim],
									  self.data_labels[0:-test_dim])
						y_pred = estimator.predict(self.data[-test_dim:])
					else:
						y_pred = estimator.fit_predict(self.data[-test_dim:])

					score = metrics.f1_score(self.data_labels[-test_dim:],
											  y_pred,
											  zero_division=0)
				except ValueError:
					pass

				if self.verbose:
					print_step("run with score: " + str(score))

				return 1 - score
		else:
			if not isinstance(self.cross_validation, bool):
				raise NotImplementedError("Not yet implemented")
