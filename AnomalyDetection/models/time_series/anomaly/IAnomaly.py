import abc
from abc import ABC


class IAnomaly(ABC):
    @abc.abstractmethod
    def anomaly_score(self, X):
        pass
