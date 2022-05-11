from __future__ import annotations
import abc
from abc import ABC
from typing import Tuple

import pandas as pd
import numpy as np


class IDataReader(ABC):
    @abc.abstractmethod
    def read(self, path: str,
             file_format: str = "csv") -> IDataReader:
        pass
    
    @abc.abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        pass

class IDataSupervised(ABC):
    @abc.abstractmethod
    def get_ground_truth(self, col_name: str) -> np.ndarray:
        pass

class IDataTrainTestSplitter(ABC):
    @abc.abstractmethod
    def train_test_split(self, train_perc: float = 0.8) -> IDataTrainTestSplitter:
        pass
    
    @abc.abstractmethod
    def get_train_test_dataframes(self) -> Tuple[pd.DataFrame]:
        pass

class IDataTrainValidTestSplitter(ABC):
    @abc.abstractmethod
    def train_valid_test_split(self, train_perc: float = 0.7,
                               valid_perc: float = 0.1) -> IDataTrainValidTestSplitter:
        pass

    @abc.abstractmethod
    def get_train_valid_test_dataframes(self) -> Tuple[pd.DataFrame]:
        pass