import abc
from abc import ABC
from typing import Callable

import numpy as np

from .IHyperparameterSearchResults import IHyperparameterSearchResults


class IHyperparameterSearch(ABC):
    """Interface describing hyperparameter searchers.
    """
    
    @abc.abstractmethod
    def search(self, x,
               y,
               objective_function: Callable[[object | np.ndarray,
                                             object | np.ndarray,
                                             np.ndarray,
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
        x : array-like of shape (n_samples, n_features) or list
            Array-like containing data over which to perform the search of the
            hyperparameters. If it is expressed as `list` it represents the
            values of the training to pass to the objective function. In that
            case, it must have the same length of y. Each element of the list
            must be an iterable (such as `Tuple`) in which the first element is
            the training x and the second element is the testing x.
        
        y : array-like of shape (n_samples, n_target) or list
            Array-like containing targets of the data on which to perform the
            search of the hyperparameters. If it is expressed as `list` it
            represents the values of the training to pass to the objective
            function. In that case, it must have the same length of y. Each
            element of the list must be an iterable (such as `Tuple`) in which
            the first element are the training labels and the second element are
            the testing labels.
        
        objective_function : Callable
            It is a function training the model and evaluating its performances.
            The first arguments are the training set, the second arguments are
            the validation set and the third argument is a dictionary of all the
            parameters of the model. The parameters must be used to instantiate
            the model. Basically, objective_function(train_data, train_labels,
            valid_data, valid_labels, parameters).
        
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        search_results : IHyperparameterSearchResults
            The results of the search.
        """
        pass
    
    @abc.abstractmethod
    def get_results(self) -> IHyperparameterSearchResults:
        """Get search results.
        
        Returns
        -------
        search_results : IHyperparameterSearchResults
            The results of the last search or the specified search at
            initialization.
        """
        pass
