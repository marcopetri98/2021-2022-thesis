import abc
from abc import ABC


class ILoader(ABC):
    """Interface for the data loaders of papers.
    
    This interface exposes the APIs used to load training, validation and
    testing datasets used by paper to train their models.
    """
    
    @abc.abstractmethod
    def get_train_valid_test(self, *args, **kwargs):
        """Gets training, validation and testing sets.
        
        Parameters
        ----------
        args
            Not used, present for inheritance change of signature.
        
        kwargs
            Not used, present for inheritance change of signature.

        Returns
        -------
        values
            Training, validation and testing in the format of the paper that
            is implemented.
        """
        pass
    