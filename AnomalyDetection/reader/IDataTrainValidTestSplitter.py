from __future__ import annotations
import abc
from abc import ABC
from typing import Tuple

import pandas as pd


class IDataTrainValidTestSplitter(ABC):
	"""Interface for all dataset readers able to perform train-valid-test split.
    """
	
	@abc.abstractmethod
	def train_valid_test_split(self, train_perc: float = 0.7,
                               valid_perc: float = 0.1) -> IDataTrainValidTestSplitter:
		"""
		
		Parameters
		----------
		train_perc : float, default=0.7
			The percentage of points of the dataset used to train the algorithm.
			
		valid_perc : float, default=0.1
			The percentage of points of the dataset used to validate the
			algorithm.

		Returns
		-------
		IDataTrainValidTestSplitter
			Instance of itself to be able to chain calls.
		"""
		pass

	@abc.abstractmethod
	def get_train_valid_test_dataframes(self) -> Tuple[pd.DataFrame]:
		"""Gets the dataframes of the training and testing.
		
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
