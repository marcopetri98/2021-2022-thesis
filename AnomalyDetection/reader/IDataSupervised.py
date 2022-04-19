import abc
from abc import ABC

import numpy as np


class IDataSupervised(ABC):
	"""Interface for all supervised dataset readers.
    """
	
	@abc.abstractmethod
	def get_ground_truth(self, col_name: str) -> np.ndarray:
		"""Gets the ground truth of the dataset.
		
		Parameters
		----------
		col_name : str
			The name of the column of the dataframe containing the ground truth.

		Returns
		-------
		ground_truth : ndarray
			The ground truth of the dataset.
		"""
		pass
