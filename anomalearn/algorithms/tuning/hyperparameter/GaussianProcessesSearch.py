import os
import time
from typing import Callable

import numpy as np
import skopt
from scipy.optimize import OptimizeResult
from skopt.callbacks import CheckpointSaver
from skopt.space import Categorical, Integer, Real

from . import HyperparameterSearchSaver, HyperparameterSearch
from ....utils.printing import print_step, print_warning


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
                 train_validation_couples: bool = False,
                 load_checkpoint: bool = False,
                 gp_kwargs: dict = None):
        super().__init__(parameter_space=parameter_space,
                         model_folder_path=model_folder_path,
                         search_filename=search_filename,
                         cross_val_generator=cross_val_generator,
                         train_validation_couples=train_validation_couples)
        
        if gp_kwargs is not None and len(set(self.__INVALID_GP_KWARGS).difference(gp_kwargs.keys())) != len(self.__INVALID_GP_KWARGS):
            raise ValueError("The following keywords of gp_minimize cannot be "
                             "passed since are automatically managed by the "
                             "class: {}".format(self.__INVALID_GP_KWARGS))
        
        self.test = 0
        self.load_checkpoint = load_checkpoint
        self.gp_kwargs = dict(gp_kwargs) if gp_kwargs is not None else {}
        
        self.__minimized_objective = None
        self.__run_opt_verbose = False
        self.__config_saver = None
    
    def _run_optimization(self, objective_function: Callable[[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              dict], float],
                          verbose: bool = False) -> OptimizeResult:
        file_path = self.model_folder_path + self.search_filename
        
        self.__minimized_objective = objective_function
        self.__run_opt_verbose = verbose
        self.__config_saver = HyperparameterSearchSaver(file_path)
        
        checkpoint_saver = CheckpointSaver(file_path + ".pkl", compress=9)
        
        callbacks = [checkpoint_saver]
        if "callback" in self.gp_kwargs.keys():
            callbacks.append(self.gp_kwargs["callback"])
            del self.gp_kwargs["callback"]
        
        if self.load_checkpoint:
            if verbose:
                print_step("Loading previous history of searches")
            
            previous_checkpoint = skopt.load(file_path + ".pkl")
            x0 = previous_checkpoint.x_iters
            y0 = previous_checkpoint.func_vals
            
            if len(x0[0]) != len(self.parameter_space):
                raise ValueError("If you are continuing a search, do not change"
                                 " the parameter space.")
            
            self._search_history = self.__config_saver.load_history()
            
            if verbose:
                print_step("Previous history has been loaded")
            
            results = skopt.gp_minimize(self._gaussian_objective,
                                        self.parameter_space,
                                        x0=x0,
                                        y0=y0,
                                        callback=callbacks,
                                        **self.gp_kwargs)
        else:
            if os.path.exists(file_path + ".pkl"):
                print_warning("There exists a checkpoint file!")
                print("Do you really want to overwrite it (you will lose it)? [y/n]: ", end="")
                response = input()
                if response.lower() == "n" or response.lower() == "no":
                    self.__run_opt_verbose = False
                    self.__minimized_objective = None
                    self.__config_saver = None
                    raise StopIteration("Stop")
            
            results = skopt.gp_minimize(self._gaussian_objective,
                                        self.parameter_space,
                                        callback=callbacks,
                                        **self.gp_kwargs)
        
        self.__run_opt_verbose = False
        self.__minimized_objective = None
        self.__config_saver = None
        
        return results
    
    def _gaussian_objective(self, args: list) -> float:
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
        start_time = time.time()
        score = self._objective_call(self.__minimized_objective,
                                     self.__run_opt_verbose,
                                     *args)
        duration = time.time() - start_time
        converted_args = self._convert_args(args)
        self._add_search_entry(score, duration, *converted_args)
        self.__config_saver(search_history=self._search_history)
        return score
    
    def _convert_args(self, args: list) -> list:
        """Converts skopt values to standard Python.
        
        Parameters
        ----------
        args : list
            The space parameters chosen by the gaussian process.

        Returns
        -------
        converted_args : list
            Args in standard Python objects.
        """
        converted_args = []
        for i in range(len(args)):
            if isinstance(self.parameter_space[i], Integer):
                converted_args.append(int(args[i]))
            elif isinstance(self.parameter_space[i], Real):
                converted_args.append(float(args[i]))
            else:
                if isinstance(args[i], str):
                    converted_args.append(args[i])
                elif np.issubdtype(args[i], np.integer):
                    converted_args.append(int(args[i]))
                elif np.issubdtype(args[i], np.floating):
                    converted_args.append(float(args[i]))
        return converted_args
