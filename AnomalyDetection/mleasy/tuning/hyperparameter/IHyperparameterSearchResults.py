import abc
from abc import ABC

import numpy as np


class IHyperparameterSearchResults(ABC):
    """Interface to represent hyperparameter search result exposed methods.
    """
    
    @abc.abstractmethod
    def get_best_score(self) -> float:
        """Gets the best score obtained on the search.
        
        Returns
        -------
        best_score : float
            The best score obtained on the search.
        """
        pass
    
    @abc.abstractmethod
    def get_best_config(self) -> dict:
        """Gets the best configuration obtained on the search.
        
        Returns
        -------
        best_config : dict
            The best configuration of the search.
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
