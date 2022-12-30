import abc
from abc import ABC
from typing import IO


class ISavable(ABC):
    """Interface for all objects whose parameters can be saved to file.
    """
    
    @abc.abstractmethod
    def save(self, file: str | IO,
             *args,
             **kwargs) -> None:
        """Saves all the parameters of the model.
        
        Parameters
        ----------
        file : str | IO
            It is the path where to save the parameters or the file-object where
            to write the parameters.

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
    def load(self, file: str | IO,
             *args,
             **kwargs) -> None:
        """Loads all the parameters of the model.
        
        Parameters
        ----------
        file : str | IO
            It is the path of the file to load the parameters or the file-object
            where to be used for reading parameters.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        pass
