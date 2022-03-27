# Python imports
import time
from typing import Union, Callable

# External imports
import numpy as np
import skopt
from scipy.optimize import OptimizeResult
from sklearn import metrics
from sklearn.utils import check_X_y
from skopt.callbacks import CheckpointSaver
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

# Project imports
from print_utils.printing import print_header, print_step


class HyperparameterSearch(BaseEstimator):
	"""HyperparameterSearch is a class used to search the hyperparameters.

	Parameters
	----------
	estimator: BaseEstimator
		The scikit-learn estimator on which to apply hyperparameter search.

	parameter_space: list,

	model_folder_path: str

	search_filename: str

	calls: int, default= 10

	initial_starts: int, default= 10

	cross_validation: bool or Callable, default=True

	train_and_test: bool default=True

	load_checkpoint: bool default=False

	verbose: bool, default=True
			States if default printing or detailed printing must be performed.
	"""

	def __init__(self, estimator: BaseEstimator,
				 parameter_space: list,
				 model_folder_path: str,
				 search_filename: str,
				 calls: int = 10,
				 initial_starts: int = 10,
				 cross_validation: Union[bool, Callable] = True,
				 train_and_test: bool = True,
				 load_checkpoint: bool = False,
				 verbose: bool = True):
		super().__init__()

		self.estimator = estimator
		self.parameter_space = parameter_space
		self.model_folder_path = model_folder_path
		self.search_filename = search_filename
		self.calls = calls
		self.initial_starts = initial_starts
		self.cross_validation = cross_validation
		self.train_and_test = train_and_test
		self.load_checkpoint = load_checkpoint
		self.verbose = verbose
		self.data = None
		self.data_labels = None

	def _build_input_dict(self, args) -> dict:
		"""Build the dictionary of the parameters.

		Parameters
		----------
		args
			The passed arguments to the optimization function.

		Returns
		-------
		parameters: dict[str]
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
		parameters: list
			The list of all the parameters used for the training process.

		Returns
		-------
		None
		"""
		parameters = []
		for parameter in self.parameter_space:
			parameters.append(parameter.name)
		parameters.append("performance")
		return parameters

	def _run_skopt_optimization(self) -> OptimizeResult:
		"""Runs the optimization search and retur the results.

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
			results = skopt.gp_minimize(self._objective,
										self.parameter_space,
										x0=x0,
										y0=y0,
										n_calls=self.calls,
										n_initial_points=self.initial_starts,
										callback=[checkpoint_saver])
		else:
			results = skopt.gp_minimize(self._objective,
										self.parameter_space,
										n_calls=self.calls,
										n_initial_points=self.initial_starts,
										callback=[checkpoint_saver])

		return results

	def _objective(self, *args) -> float:
		"""The function implementing the loss to minimize.

		Parameters
		----------
		args: list
			The list of all the parameters passed to the function to be able
			to run the algorithm.

		Returns
		-------
		function_value: float
			The value of the computed objective function.
		"""
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

	def fit(self, X, y=None) -> None:
		"""Performs the hyperparameter search.

		Parameters
		----------
		X: array-like of shape (n_samples, n_features)
			The training data.
		y: array-like of shape (n_samples, n_labels)
			The labels for the training data. Unused in case of unsupervised
			learning methods.

		Returns
		-------
		None
		"""
		check_X_y(X, y)
		X = np.array(X)
		y = np.array(y)

		start_time = time.time()
		self.data, self.data_labels = X, y

		if self.verbose:
			print_header("Starting the hyperparameter search")

		res = self._run_skopt_optimization()
		tries = [self._create_first_row()]
		for i in range(len(res.x_iters)):
			elem = res.x_iters[i].copy()
			elem.append(res.func_vals[i])
			tries.append(elem)
		tries = np.array(tries, dtype=object)
		np.save(self.model_folder_path + self.search_filename, tries)

		self.data, self.data_labels = None, None
		if self.verbose:
			print_step("Search lasted for: " + str(time.time() - start_time))
			print_header("Hyperparameter search ended")

	def print_search(self) -> None:
		"""Print the results of the search represented by this object.

		Returns
		-------
		None
		"""
		tries = np.load(self.model_folder_path + self.search_filename + ".npy",
						allow_pickle=True)
		indices = (np.argsort(tries[1:, -1]) + 1).tolist()
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
