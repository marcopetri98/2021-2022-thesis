from __future__ import annotations
import abc
from abc import ABC

import pandas as pd


class IDataReader(ABC):
    """Interface for all dataset readers in the repository.
    """
    
    @abc.abstractmethod
    def read(self, path: str,
             file_format: str = "csv",
             verbose: bool = True,
             *args,
             **kwargs) -> IDataReader:
        """Reads a dataset and returns its numpy version.
        
        Parameters
        ----------
        path : str
            It is a string representing the location on disk of the dataset to
            read.
        
        file_format : str, default="csv"
            It is the format in which the dataset is stored.
            
        verbose : bool, default=True
            States if detailed printing must be done while reading the dataset.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        IDataReader
            Instance of itself to be able to chain calls.
        """
        pass
    
    @abc.abstractmethod
    def get_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        """Gets the dataframe of the dataset previously read.
        
        Parameters
        ----------
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.
        
        Returns
        -------
        dataframe : DataFrame
            The dataframe of the dataset.
        """
        pass
