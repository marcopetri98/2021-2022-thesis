import abc
from abc import ABC

import numpy as np


class ICluster(ABC):
	"""Interface identifying a machine learning algorithm performing clustering.
    """
	
	@abc.abstractmethod
	def cluster(self, X) -> np.ndarray:
		pass
