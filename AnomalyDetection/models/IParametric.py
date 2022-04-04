import abc
from abc import ABC


class IParametric(ABC):
	"""Interface identifying a machine learning parametric model.
    """
	
	@abc.abstractmethod
	def fit(self, X, y=None) -> None:
		"""Fits the model to the given training data.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training data representing containing the features.

		y : array-like of shape (n_samples, n_label_features)
			The target for the training data which may be used by either
			classification or regression models.

		Returns
		-------
		None
		"""
		pass
