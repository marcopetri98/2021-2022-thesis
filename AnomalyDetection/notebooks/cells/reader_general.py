from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd


class TimeSeriesReader(IDataReader,
                       IDataSupervised,
                       IDataTrainTestSplitter,
                       IDataTrainValidTestSplitter):
    ACCEPTED_FORMATS = ["csv"]
    
    def __init__(self):
        super().__init__()
        
        self.path : str = ""
        self.format : str = ""
        self.dataset : pd.DataFrame = None
        self.train_frame : pd.DataFrame = None
        self.valid_frame : pd.DataFrame = None
        self.test_frame : pd.DataFrame = None
    
    def read(self, path: str,
             file_format: str = "csv") -> TimeSeriesReader:
        if file_format not in self.ACCEPTED_FORMATS:
            raise ValueError("The file format must be one of %s" %
                             self.ACCEPTED_FORMATS)
        elif path == "":
            raise ValueError("The path cannot be empty")
        
        self.path = path
        self.format = file_format
        
        self.dataset = pd.read_csv(self.path)
        
        return self
    
    def train_test_split(self, train_perc: float = 0.8) -> TimeSeriesReader:
        check_not_default_attributes(self, {"dataset": None})
        
        if not 0 < train_perc < 1:
            raise ValueError("The training percentage must lie in (0,1) range.")
        
        num_of_test = int((1 - train_perc) * self.dataset.shape[0])
        self.train_frame = self.dataset[0:-num_of_test]
        self.test_frame = self.dataset[-num_of_test:]
        
        return self
    
    def train_valid_test_split(self, train_perc: float = 0.7,
                               valid_perc: float = 0.1) -> TimeSeriesReader:
        check_not_default_attributes(self, {"dataset": None})
        
        if not 0 < train_perc < 1 or not 0 < valid_perc < 1:
            raise ValueError("Training and validation percentages must lie in "
                             "(0,1) range.")
        elif train_perc + valid_perc >= 1:
            raise ValueError("Training and validation must be less than all the "
                             "dataset, i.e., their sum lies in (0,1).")
        
        num_of_not_train = int((1 - train_perc) * self.dataset.shape[0])
        num_of_test = int((1 - train_perc - valid_perc) * self.dataset.shape[0])
        self.train_frame = self.dataset[0:-num_of_not_train]
        self.valid_frame = self.dataset[-num_of_not_train:-num_of_test]
        self.test_frame = self.dataset[-num_of_test:]
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        check_not_default_attributes(self, {"dataset": None})
        return self.dataset.copy()
    
    def get_train_test_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        check_not_default_attributes(self, {"train_frame": None,
                                            "test_frame": None})
        return self.train_frame.copy(), self.test_frame.copy()
    
    def get_train_valid_test_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        check_not_default_attributes(self, {"train_frame": None,
                                            "valid_frame": None,
                                            "test_frame": None})
        return self.train_frame.copy(), self.valid_frame.copy(), self.test_frame.copy()
    
    def get_ground_truth(self, col_name: str) -> np.ndarray:
        check_not_default_attributes(self, {"dataset": None})
        
        if col_name not in self.dataset.columns:
            raise ValueError("The column specified does not exist")
        
        targets = self.dataset[col_name]
        return np.array(targets)
