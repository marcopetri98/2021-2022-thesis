from __future__ import annotations
import abc
from abc import ABC
from typing import Tuple

import pandas as pd


class IDataTrainTestSplitter(ABC):
	"""Interface for all dataset readers able to perform train-test split.
    """
	
	@abc.abstractmethod
	def train_test_split(self, train: float | int = 0.8) -> IDataTrainTestSplitter:
		"""Splits the dataset in training and test.
		
		Parameters
		----------
		train : float or int, default=0.8
			The percentage of points of the dataset used to train the algorithm
			if `float`, the number of training data points from the start to the
			end if `int`.

		Returns
		-------
		IDataTrainTestSplitter
			Instance of itself to be able to chain calls.
		"""
		pass
	
	@abc.abstractmethod
	def get_train_test_dataframes(self) -> Tuple[pd.DataFrame]:
		"""Gets the dataframes of the training and testing.
		
		Returns
		-------
		train_df : DataFrame
			The training dataframe of the dataset given the train-test split.
			
		test_df : DataFrame
			The test dataframe of the dataset given the train-test split.
		"""
		pass
