import abc
from abc import ABC


class IParametric(ABC):
	"""Interface identifying a machine learning parametric model.
    """
	
	@abc.abstractmethod
	def fit(self, X, y=None) -> None:
		pass
