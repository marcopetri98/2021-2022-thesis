import abc
from abc import ABC

import numpy as np


class IPredictor(ABC):
	"""Interface identifying a machine learning predictor.
    """
	
	@abc.abstractmethod
	def predict(self) -> np.ndarray:
		pass
