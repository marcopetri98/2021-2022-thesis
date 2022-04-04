import abc

import numpy as np

from models.IRegressor import IRegressor


class IAnomalyRegressor(IRegressor):
	"""Interface identifying a machine learning algorithm giving anomaly scores.
    """
	
	@abc.abstractmethod
	def anomaly_score(self, X) -> np.ndarray:
		pass
