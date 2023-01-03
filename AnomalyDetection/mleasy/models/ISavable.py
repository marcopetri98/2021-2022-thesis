import abc
from abc import ABC
from typing import IO


class ISavable(ABC):
    """Interface for all objects whose parameters can be saved to file.
    """
    
    @abc.abstractmethod
    def save(self, file_path: str,
             *args,
             **kwargs) -> None:
        """Saves all the parameters of the model.
        
        Parameters
        ----------
        file_path : str
            It is the path where to save the model. All the file paths are
            extended to be saved in a file with extensions ".pickle", i.e., if
            the path does not end with ".pickle", the string ".pickle" will be
            added to it.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        pass
    
    @abc.abstractmethod
    def load(self, file_path: str,
             *args,
             **kwargs) -> None:
        """Loads all the parameters of the model.
        
        Parameters
        ----------
        file_path : str
            It is the path of the file to read containing the model.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        pass
