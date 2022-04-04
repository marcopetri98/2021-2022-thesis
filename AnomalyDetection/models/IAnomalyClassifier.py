from abc import ABC

from models.IClassifier import IClassifier


class IAnomalyClassifier(IClassifier, ABC):
	"""Interface identifying a machine learning anomaly classifier.
    """
