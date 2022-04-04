import abc
from abc import ABC

import numpy as np


class IClassifier(ABC):
	"""Interface identifying a machine learning classifier.
    """
	
	@abc.abstractmethod
	def classify(self, X) -> np.ndarray:
		pass
