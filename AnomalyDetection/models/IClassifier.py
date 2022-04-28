import abc
from abc import ABC

import numpy as np


class IClassifier(ABC):
	"""Interface identifying a machine learning classifier.
    """
	
	@abc.abstractmethod
	def classify(self, X, *args, **kwargs) -> np.ndarray:
		"""Computes the labels for the given points.
		
		If a point has 1 as label it is classified as anomaly while if it has
		0 as label it is classified as normal. **Please, note that** if the
		model is parametric (inherits from IParametric) you must first perform
		fit on training data.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The points for which we must compute the anomaly score.
			
		*args
			Not used, present to allow multiple inheritance and signature change.
			
		*kwargs
			Not used, present to allow multiple inheritance and signature change.

		Returns
		-------
		anomaly_labels : ndarray of shape (n_samples,)
			The scores of the points.
		"""
		pass
