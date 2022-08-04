import abc
from abc import ABC

import numpy as np


class IPredictor(ABC):
	"""Interface identifying a machine learning predictor.
    """
	
	@abc.abstractmethod
	def predict(self, X, *args, **kwargs) -> np.ndarray:
		"""
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training data representing containing the features.
			
		args
			Not used, present to allow multiple inheritance and signature change.
			
		kwargs
			Not used, present to allow multiple inheritance and signature change.

		Returns
		-------
		predictions : ndarray of shape (n_samples, n_prediction)
			The prediction for each of the samples given in input.
		"""
		pass
