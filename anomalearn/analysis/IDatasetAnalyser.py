import abc
from abc import ABC


class IDatasetAnalyser(ABC):
    """Interface for dataset analysers.
    """
    
    @abc.abstractmethod
    def num_samples(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns
        -------
        num_samples : int
            The number of samples in the dataset.
        """
        pass
