from __future__ import annotations
import abc
from abc import ABC
from typing import Tuple

import pandas as pd


class IDataTrainValidTestSplitter(ABC):
    """Interface for all dataset readers able to perform train-valid-test split.
    """
    
    @abc.abstractmethod
    def train_valid_test_split(self, train: float | int = 0.7,
                               valid: float | int = 0.1,
                               *args,
                               **kwargs) -> IDataTrainValidTestSplitter:
        """Split the dataset into training, validation and testing.
        
        Parameters
        ----------
        train : float or int, default=0.7
            The percentage of points of the dataset used to train the algorithm
            if `float`, the number of training data points from the start to the
            end if `int`.
            
        valid : float or int, default=0.1
            The percentage of points of the dataset used to validate the
            algorithm if `float`, the number of validation data points from the
            end of the training if `int`.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        IDataTrainValidTestSplitter
            Instance of itself to be able to chain calls.
        """
        pass
    
    @abc.abstractmethod
    def get_train_valid_test_dataframes(self, *args,
                                        **kwargs) -> Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.DataFrame]:
        """Gets the dataframes of the training and testing.
        
        Parameters
        ----------
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.
        
        Returns
        -------
        train_df : DataFrame
            The training dataframe of the dataset given the train split.
            
        valid_df : DataFrame
            The validation dataframe of the dataset given the valid split.
            
        test_df : DataFrame
            The test dataframe of the dataset given the test split.
        """
        pass
