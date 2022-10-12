from __future__ import annotations
import abc
from abc import ABC

import pandas as pd


class IDataMultipleReader(ABC):
    """Interface for a dataset reader able to read multiple dataset at the same time.
    """
    
    @abc.abstractmethod
    def read_multiple(self, paths: list[str],
                      files_format: str = "csv",
                      verbose: bool = True,
                      *args,
                      **kwargs) -> IDataMultipleReader:
        """Reads multiple datasets at the same time.
        
        Parameters
        ----------
        paths : str
            It is a list of strings representing the paths at which the dataset
            to read are stored.
        
        files_format : str, default="csv"
            It is the format in which the datasets are stored.
        
        verbose : bool, default=True
            States if detailed printing must be done while reading the dataset.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        IDataMultipleReader
            Instance of itself to be able to chain calls.
        """
        pass
    
    @abc.abstractmethod
    def get_all_dataframes(self) -> list[pd.DataFrame]:
        """Gets all the dataframes previously read.
        
        Returns
        -------
        dataframes : list[DataFrame]
            All the dataframes in the same order specified with read_multiple.
        """
        pass
    
    @abc.abstractmethod
    def get_ith_dataframe(self, pos: int) -> pd.DataFrame:
        """Get the ith dataframe in all the read ones.
        
        Parameters
        ----------
        pos : int
            The position of the dataframe to be read.

        Returns
        -------
        dataframe : DataFrame
            The ith dataframe in the order specified with read_multiple.
        """
        pass
    