import abc
from abc import ABC

import numpy as np


class ICluster(ABC):
	"""Interface identifying a machine learning algorithm performing clustering.
    """
	
	@abc.abstractmethod
	def cluster(self, X, *args, **kwargs) -> np.ndarray:
		"""Clusters the data.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The data to be clustered.
			
		args
			Not used, present to allow multiple inheritance and signature change.
			
		kwargs
			Not used, present to allow multiple inheritance and signature change.

		Returns
		-------
		clusters : ndarray of shape (n_samples, n_clusters)
			An array identifying the cluster at which each point is associated.
		"""
		pass
