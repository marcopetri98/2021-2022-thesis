from __future__ import annotations

import pandas as pd

from mleasy.input_validation import check_not_default_attributes
from mleasy.reader import IDataMultipleReader
from mleasy.reader.time_series import TSReader
from mleasy.utils import print_header, print_step


class TSMultipleReader(TSReader, IDataMultipleReader):
    """A time series reader able to read multiple data series at the same time"""
    
    def __init__(self):
        super().__init__()
        
        self.all_dataframes : list = None
        
    def read_multiple(self, paths: list[str],
                      files_format: str = "csv",
                      verbose: bool = True,
                      *args,
                      **kwargs) -> TSMultipleReader:
        if verbose:
            print_header("Start reading all datasets")
            
        self.all_dataframes = list()
        for idx, path in enumerate(paths):
            if verbose:
                print_step("Start to read the {}th dataset".format(idx))
                
            self.read(path, files_format, False)
            self.all_dataframes.append(self.dataset)
            
            if verbose:
                print_step("Finished to read the {}th dataset".format(idx))
        
        if verbose:
            print_header("All datasets read")
            
        return self
    
    def select_dataframe(self, pos: int) -> None:
        """Selects the dataset to be used.
        
        Parameters
        ----------
        pos : int
            The dataset to select to perform single dataset operations.

        Returns
        -------
        None
        """
        if pos <= 0 or pos >= len(self.all_dataframes):
            raise ValueError("There are {} dataframes, specify a valid index".format(len(self.all_dataframes)))
        
        self.dataset = self.all_dataframes[pos]
    
    def get_all_dataframes(self) -> list[pd.DataFrame]:
        check_not_default_attributes(self, {"all_dataframes": None})
        return self.all_dataframes.copy()
    
    def get_ith_dataframe(self, pos: int) -> pd.DataFrame:
        check_not_default_attributes(self, {"all_dataframes": None})
        
        if pos <= 0 or pos >= len(self.all_dataframes):
            raise ValueError("There are {} dataframes, specify a valid index".format(len(self.all_dataframes)))
        
        return self.all_dataframes[pos].copy()
