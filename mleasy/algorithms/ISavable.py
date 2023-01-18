import abc
from abc import ABC


class ISavable(ABC):
    """Interface for all objects whose parameters can be saved to file.
    """
    
    @abc.abstractmethod
    def save(self, path: str,
             *args,
             **kwargs) -> None:
        """Saves the objects state.
        
        Parameters
        ----------
        path : str
            It is the path of the folder in which the object will be saved.

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
    def load(self, path: str,
             *args,
             **kwargs) -> None:
        """Loads all the parameters of the model.
        
        Parameters
        ----------
        path : str
            It is the path of the directory in which the object has been saved.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        pass
