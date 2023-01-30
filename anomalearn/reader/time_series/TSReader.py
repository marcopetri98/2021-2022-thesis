from __future__ import annotations

import os.path

import pandas as pd

from .. import IDataReader
from ...input_validation.attribute_checks import check_not_default_attributes
from ...utils.printing import print_header, print_step


class TSReader(IDataReader):
    """A reader of time series datasets."""
    ACCEPTED_FORMATS = ["csv", "json", "xml"]
    
    def __init__(self):
        super().__init__()
        
        self.dataset: pd.DataFrame | None = None
    
    def read(self, path,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> TSReader:
        if not os.path.isfile(path):
            raise ValueError(f"The file path \"{path}\" you are trying to read "
                             "does not exists.")
        elif pandas_args is not None and not isinstance(pandas_args, dict):
            raise TypeError("pandas_args must be None or a dict")
        
        if verbose:
            print_header("Start reading dataset")
            print_step("Start to read csv using pandas")

        pandas_args = pandas_args if pandas_args is not None else {}

        match file_format:
            case "csv":
                self.dataset = pd.read_csv(path, **pandas_args)

            case "json":
                self.dataset = pd.read_json(path, **pandas_args)

            case "xml":
                self.dataset = pd.read_xml(path, **pandas_args)

            case _:
                raise NotImplementedError("The dataset format is not supported,"
                                          " the accepted formats are "
                                          f"{self.ACCEPTED_FORMATS}")
        
        if verbose:
            print_step("Ended pandas reading")
            print_header("Ended dataset reading")
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        check_not_default_attributes(self, {"dataset": None})
        return self.dataset.copy()
