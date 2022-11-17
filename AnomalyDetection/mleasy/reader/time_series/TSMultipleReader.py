from __future__ import annotations

import pandas as pd

from mleasy.input_validation import check_not_default_attributes
from mleasy.reader import IDataMultipleReader
from mleasy.reader.time_series import TSReader
from mleasy.utils import print_header, print_step


class TSMultipleReader(TSReader, IDataMultipleReader):
    """A time series reader able to read multiple data series at the same time.

    This class is able to read multiple data series in a single call. The time
    series might be from the same dataset as well they might be from different
    datasets.
    """
    
    def __init__(self):
        super().__init__()
        
        self.all_dataframes: list | None = None
        
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
                print_step(f"Start to read the {idx}th dataset")
                
            self.read(path, files_format, False)
            self.all_dataframes.append(self.dataset)
            
            if verbose:
                print_step(f"Finished to read the {idx}th dataset")

        self.select_dataframe(pos=0)

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
            raise IndexError(f"There are {len(self.all_dataframes)} dataframes")
        
        self.dataset = self.all_dataframes[pos]
    
    def get_all_dataframes(self) -> list[pd.DataFrame]:
        check_not_default_attributes(self, {"all_dataframes": None})
        return self.all_dataframes.copy()
    
    def get_ith_dataframe(self, pos: int,
                          *args,
                          **kwargs) -> pd.DataFrame:
        check_not_default_attributes(self, {"all_dataframes": None})
        
        if pos <= 0 or pos >= len(self.all_dataframes):
            raise IndexError(f"There are {len(self.all_dataframes)} dataframes")
        
        return self.all_dataframes[pos].copy()
