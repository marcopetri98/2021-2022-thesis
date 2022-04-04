import abc
from abc import ABC

import numpy as np


class IRegressor(ABC):
	"""Interface identifying a machine learning regressor.
    """
	
	@abc.abstractmethod
	def regress(self, X) -> np.ndarray:
		pass
