from __future__ import annotations

import os.path

import pandas as pd

from mleasy.input_validation.attribute_checks import check_not_default_attributes
from mleasy.reader import IDataReader
from mleasy.utils.printing import print_header, print_step


class TSReader(IDataReader):
    """A reader of time series datasets."""
    ACCEPTED_FORMATS = ["csv"]
    
    def __init__(self):
        super().__init__()
        
        self.dataset: pd.DataFrame | None = None
    
    def read(self, path: str,
             file_format: str = "csv",
             verbose: bool = True,
             *args,
             **kwargs) -> TSReader:
        if file_format not in self.ACCEPTED_FORMATS:
            raise ValueError(f"The file format must be one of {self.ACCEPTED_FORMATS}")
        elif not os.path.isfile(path):
            raise ValueError(f"The file path \"{path}\" you are trying to read does not exists.")
        
        if verbose:
            print_header("Start reading dataset")
            print_step("Start to read csv using pandas")
        
        self.dataset = pd.read_csv(path)
        
        if verbose:
            print_step("Ended pandas reading")
            print_header("Ended dataset reading")
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        check_not_default_attributes(self, {"dataset": None})
        return self.dataset.copy()
