import abc
import time
from abc import ABC
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import check_X_y

from .HyperparameterSearchResults import HyperparameterSearchResults
from .IHyperparameterSearch import IHyperparameterSearch
from .IHyperparameterSearchResults import IHyperparameterSearchResults
from mleasy.utils.json import load_py_json, save_py_json
from mleasy.utils.lists import all_indices
from mleasy.utils.printing import print_header, print_step


class HyperparameterSearch(IHyperparameterSearch, ABC):
    """Abstract class implementing some hyperparameter search methods.
    
    Parameters
    ----------
    parameter_space : list
        It is the space of the parameters to explore to perform hyperparameter
        search. This object must be a list of skopt space objects. To be precise,
        this function uses scikit-optimize as core library of implementation.

    model_folder_path : str
        It is the folder where the search results must be saved.

    search_filename : str
        It is the filename of the file where to store the results of the search.
        
    cross_val_generator : object, default=None
        It is the cross validation generator returning a train/test generator.
        If nothing is passed, standard K Fold Cross validation is used. Moreover,
        the cross validation generator must provide the same interface provided
        by scikit-learn cross validation generators.
        
    train_validation_couples : bool, default=False
        If True
    """
    _DURATION = "Duration"
    _SCORE = "Score"
    _EXT = ".search"
    
    def __init__(self, parameter_space: list,
                 model_folder_path: str,
                 search_filename: str,
                 cross_val_generator: object = None,
                 train_validation_couples: bool = False):
        super().__init__()
        
        self.parameter_space = parameter_space
        self.model_folder_path = model_folder_path
        self.search_filename = search_filename
        self.cross_val_generator = cross_val_generator
        self.train_validation_couples = train_validation_couples
        
        self._data = None
        self._data_labels = None
        self._search_history = None
    
    def search(self, x,
               y,
               objective_function: Callable[[object | np.ndarray,
                                             object | np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             dict], float],
               verbose: bool = False,
               *args,
               **kwargs) -> IHyperparameterSearchResults:
        if not self.train_validation_couples:
            check_X_y(x, y)
        else:
            if not isinstance(x, list) or not isinstance(y, list):
                raise TypeError("x and y must be lists of the same length")
            else:
                for el1, el2 in zip(x, y):
                    if not isinstance(el1, tuple) or not isinstance(el2, tuple):
                        raise TypeError("x and y must contain tuples of "
                                        "dimension 2")
                    elif len(el1) != 2 or len(el2) != 2:
                        raise ValueError("tuples must have dimension 2, please "
                                         "read the docs")
        
        if not self.train_validation_couples:
            x = np.array(x)
            y = np.array(y)
        
        start_time = time.time()
        self._data, self._data_labels = x, y
        
        if verbose:
            print_header("Starting the hyperparameter search")
        
        try:
            self._run_optimization(objective_function, verbose=verbose)
            
            final_history = self._create_result_history()
            save_py_json(final_history, self.model_folder_path + self.search_filename + self._EXT)
            
            results = HyperparameterSearchResults(final_history)
        except StopIteration as e:
            if e.value != "Stop":
                raise e
            
            results = HyperparameterSearchResults([])
        
        self._data, self._data_labels, self._search_history = None, None, None
        if verbose:
            print_step("Search lasted for: " + str(time.time() - start_time))
            print_header("Hyperparameter search ended")
        
        return results
    
    def get_results(self) -> IHyperparameterSearchResults:
        history = load_py_json(self.model_folder_path + self.search_filename + self._EXT)
        history = history if history is not None else []
        
        return HyperparameterSearchResults(history)
    
    @abc.abstractmethod
    def _run_optimization(self, objective_function: Callable[[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              dict], float],
                          verbose: bool = False):
        """Runs the optimization search.
        
        Parameters
        ----------
        objective_function : Callable
            The objective function to minimize.
            
        Returns
        -------
        None
        
        Raises
        ------
        StopIteration
            If the search must be aborted due to any problem.
            
        Notes
        -----
        See :meth:`~tuning.hyperparameter.HyperparameterSearch.search` for more
        details about objective_function.
        """
        pass
    
    def _objective_call(self, objective_function: Callable[[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            dict], float],
                        verbose: bool = False,
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
        params = self._build_input_dict(*args)
        
        if verbose:
            print_step("Trying the configuration with: ", params)
        
        score = 0
        
        if not self.train_validation_couples:
            # The data are passed as numpy array and must be cross validated
            if self.cross_val_generator is None:
                self.cross_val_generator = KFold(n_splits=5, random_state=98)
            
            for train, test in self.cross_val_generator.split(self._data,
                                                              self._data_labels):
                x_train, x_test = self._data[train], self._data[test]
                y_train, y_test = self._data_labels[train], self._data_labels[test]
                obj = objective_function(x_train, y_train, x_test, y_test, params)
                score += obj
            
            score /= self.cross_val_generator.get_n_splits()
        else:
            # The training and testing couples are directly passed
            for data, labels in zip(self._data, self._data_labels):
                x_train, x_test = data[0], data[1]
                y_train, y_test = labels[0], labels[1]
                obj = objective_function(x_train, y_train, x_test, y_test, params)
                score += obj
                
            score /= len(self._data)
        
        if verbose:
            print_step("The configuration has a score of ", score)
        
        return score
    
    def _add_search_entry(self, score,
                          duration,
                          *args) -> None:
        """It adds an entry to the search history.
        
        Parameters
        ----------
        score : float
            The score of this configuration.
            
        duration : float
            The duration of trying this configuration.
        
        args
            The passed arguments to the optimization function.

        Returns
        -------
        None
        """
        params = self._build_input_dict(*args)
        
        # Since tuples are saved to json arrays and json arrays are always
        # converted to lists, I need to convert any tuple to a list to be able
        # to check the identity since results may be loaded from a checkpoint
        for key, item in params.items():
            if isinstance(item, tuple):
                params[key] = list(item)
        
        if self._search_history is None:
            self._search_history = {self._SCORE: [score],
                                    self._DURATION : [duration]}
            
            for key, value in params.items():
                self._search_history[key] = [value]
        else:
            self._search_history[self._SCORE].append(score)
            self._search_history[self._DURATION].append(duration)
            
            common_keys = set(self._search_history.keys()).intersection(set(params.keys()))
            only_history_keys = set(self._search_history.keys()).difference(set(params.keys())).difference({self._SCORE, self._DURATION})
            only_params_keys = set(params.keys()).difference(set(self._search_history.keys()))
            
            for key in common_keys:
                self._search_history[key].append(params[key])
            
            for key in only_history_keys:
                self._search_history[key].append(None)
            
            for key in only_params_keys:
                self._search_history[key] = [None] * len(self._search_history[self._SCORE])
                self._search_history[key][-1] = params[key]
    
    def _find_history_entry(self, *args) -> bool:
        """Search if a configuration has just been tried.
        
        Parameters
        ----------
        args
            The passed arguments to the optimization function.

        Returns
        -------
        is_present : bool
            True if the configuration has been already tried, False otherwise.
        """
        params = self._build_input_dict(*args)
        found = False
        
        # Since tuples are saved to json arrays and json arrays are always
        # converted to lists, I need to convert any tuple to a list to be able
        # to check the identity since results may be loaded from a checkpoint
        for key, item in params.items():
            if isinstance(item, tuple):
                params[key] = list(item)
        
        if self._search_history is not None:
            only_params_keys = set(params.keys()).difference(set(self._search_history.keys()))
            
            if len(only_params_keys) == 0:
                # All keys passed are keys of the history, it might be a
                # configuration already tried.
                config = None
                for key in params.keys():
                    curr = set(all_indices(self._search_history[key], params[key]))
                    
                    if config is None:
                        # For the first key I search the identical values
                        config = curr
                    elif len(config) == 0:
                        # Since config has no elements, config has not been tried
                        # yet
                        break
                    else:
                        # Search another key and compare indexes
                        config = config.intersection(curr)
                
                if len(config) > 0:
                    found = True
        
        return found
    
    def _build_input_dict(self, *args) -> dict:
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
        parameters = [self._SCORE, self._DURATION]
        for parameter in self.parameter_space:
            parameters.append(parameter.name)
        return parameters
    
    def _create_result_history(self) -> list:
        """Creates the search result list from history.
        
        Returns
        -------
        search : list
            The results of the search with as first row the names of the
            parameters of the model.
        """
        keys = [key
                for key in self._search_history.keys()]
        tries = [[self._search_history[key][i]
                  for key in self._search_history.keys()]
                 for i in range(len(self._search_history[self._SCORE]))]
        final_history = [keys]
        
        for try_ in tries:
            final_history.append(try_)
        
        return final_history
