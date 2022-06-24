from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd

from input_validation.attribute_checks import check_not_default_attributes
from reader.IDataReader import IDataReader
from reader.IDataSupervised import IDataSupervised
from reader.IDataTrainTestSplitter import IDataTrainTestSplitter
from reader.IDataTrainValidTestSplitter import IDataTrainValidTestSplitter
from utils.printing import print_header, print_step


class TSReader(IDataReader,
               IDataSupervised,
               IDataTrainTestSplitter,
               IDataTrainValidTestSplitter):
    """A reader of time series datasets."""
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
             file_format: str = "csv",
             verbose: bool = True,
             *args,
             **kwargs) -> TSReader:
        if file_format not in self.ACCEPTED_FORMATS:
            raise ValueError("The file format must be one of %s" %
                             self.ACCEPTED_FORMATS)
        elif path == "":
            raise ValueError("The path cannot be empty")
        
        if verbose:
            print_header("Start reading dataset")
        
        self.path = path
        self.format = file_format
        
        if verbose:
            print_step("Start to read csv using pandas")
        
        self.dataset = pd.read_csv(self.path)
        
        if verbose:
            print_step("Ended pandas reading")
            print_header("Ended dataset reading")
        
        return self
    
    def train_test_split(self, train: float | int = 0.8) -> TSReader:
        check_not_default_attributes(self, {"dataset": None})
        
        if isinstance(train, float):
            if not 0 < train < 1:
                raise ValueError("The training percentage must lie in (0,1) range.")
        elif isinstance(train, int):
            if train >= self.dataset.shape[0]:
                raise ValueError("The training must be less than the whole "
                                 "dataset.")
        else:
            raise TypeError("Argument train must be float or int")
        
        if isinstance(train, float):
            num_of_test = int((1 - train) * self.dataset.shape[0])
            self.train_frame = self.dataset[0:-num_of_test]
            self.test_frame = self.dataset[-num_of_test:]
        else:
            self.train_frame = self.dataset[0:train]
            self.test_frame = self.dataset[train:]
        
        return self
    
    def train_valid_test_split(self, train: float | int = 0.7,
                               valid: float | int = 0.1) -> TSReader:
        check_not_default_attributes(self, {"dataset": None})
        
        if isinstance(train, float) and isinstance(valid, float):
            if not 0 < train < 1 or not 0 < valid < 1:
                raise ValueError("Training and validation percentages must lie "
                                 "in (0,1) range.")
            elif train + valid >= 1:
                raise ValueError("Training and validation must be less than all"
                                 " the dataset, i.e., their sum lies in (0,1).")
        elif isinstance(train, int) and isinstance(valid, int):
            if train + valid >= self.dataset.shape[0]:
                raise ValueError("Training and validation must be lower than "
                                 "the dimension of the whole dataset.")
        else:
            raise TypeError("Either both training and validation are float or "
                            "both are int. No other configurations are allowed")
        
        if isinstance(train, float):
            num_of_not_train = int((1 - train) * self.dataset.shape[0])
            num_of_test = int((1 - train - valid) * self.dataset.shape[0])
            self.train_frame = self.dataset[0:-num_of_not_train]
            self.valid_frame = self.dataset[-num_of_not_train:-num_of_test]
            self.test_frame = self.dataset[-num_of_test:]
        else:
            self.train_frame = self.dataset[0:train]
            self.valid_frame = self.dataset[train:valid]
            self.test_frame = self.dataset[valid:]
        
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
