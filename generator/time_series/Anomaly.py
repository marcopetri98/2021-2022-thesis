from statistics import mean
from typing import Tuple, Callable
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class AnomalyPrints(object):
    ERROR_IMPL_ANOMALY = "Anomaly type not implemented"
    ERROR_TOO_SHORT = "A collective anomaly must have at least 2 points"
    ERROR_PARAM_MATCH = "The given parameters does not match with the type"


class Anomaly(object):
    """A class representing the implementation of an anomaly type

    Parameters
    ----------
    anomaly_type : str
         The type of anomaly.

    collective_type : str
        Type of collective anomaly in case it is collective.

    anomaly_length : int
        The length of the anomaly.

    anomaly_params : list[float] or Callable[[int, ndarray, tuple[int]], list[float]]
        The list of the parameters of the anomaly type.

    Notes
    -----
    Point anomalies take as parameters a series of offsets. The series of offset
    is the set of values that will be added to the real dataset value.

    Collective anomalies take as parameters the offset/constant values (same as
    for a point anomalies, i.e., a list of the possible offsets or constants) or
    a callable function taking the number of points to generate as input (int),
    the dataset and the position at which generating them and producing as
    output a list of values to be added to the time series. The mean takes as
    input the number of previous points to consider computing the mean at which
    the collective anomaly will be constant.
    """

    ALLOWED_TYPES = ["point",
                     "collective"]
    ALLOWED_COLLECTIVE = ["offset",
                          "mean",
                          "function",
                          "constant"]

    def __init__(self, anomaly_type: str,
                 collective_type: str = None,
                 anomaly_length: int = 1,
                 anomaly_params: list[float | Callable[[int, np.ndarray, int | Tuple[int]], list[float]]] = None):
        if anomaly_type not in self.ALLOWED_TYPES:
            raise ValueError(AnomalyPrints.ERROR_IMPL_ANOMALY)
        elif collective_type is not None and collective_type not in self.ALLOWED_COLLECTIVE:
            raise ValueError(AnomalyPrints.ERROR_IMPL_ANOMALY)
        elif anomaly_type == "collective" and anomaly_length < 2:
            raise ValueError(AnomalyPrints.ERROR_TOO_SHORT)
        elif not Anomaly.__check_anomaly_params(anomaly_type,
                                                anomaly_params,
                                                collective_type):
            raise ValueError(AnomalyPrints.ERROR_PARAM_MATCH)

        super().__init__()
        self.anomaly_type = anomaly_type
        self.collective_type = collective_type
        self.anomaly_length = anomaly_length
        self.anomaly_params = anomaly_params

    def compute_anomaly(self, dataset: np.ndarray,
                        position: int | Tuple[int]) -> float | list[float]:
        """Compute the anomaly in the position

        Parameters
        ----------
        dataset : ndarray of shape (n,) or (n, n_var)
            The dataset on which we need to add the anomaly.

        position : int or tuple[int]
            The position on which we want to add the anomaly. If the dataset is
            multivariate, the tuple has two elements. The first element of the
            tuple is the position of the anomaly in the time series, the second
            is the dimension of the time series in which the anomaly must be
            added.
        """
        if dataset.ndim == 1:
            is_univariate = True
        else:
            if dataset.shape[1] == 1:
                is_univariate = True
            else:
                is_univariate = False

            if dataset.ndim != 2:
                raise ValueError("A dataset must have shape (n,) or (n,n_var)")

        if not is_univariate:
            if not isinstance(position, tuple):
                raise TypeError("The position must be a tuple for multivariate"
                                " time series.")
            elif len(position) != 2:
                raise ValueError("Position must be a tuple of length 2 for "
                                 "multivariate time series")
            elif not 0 <= position[0] <= dataset.shape[0] - 1:
                raise ValueError("The position of the anomaly must be inside "
                                 "the dataset (0 <= pos < dataset.length)")
            elif not 0 <= position[1] <= dataset.shape[1] - 1:
                raise ValueError("The dimension in which the anomaly must be "
                                 "valid")
        else:
            if not isinstance(position, int):
                raise TypeError("position must be an int for univariate")
            elif not 0 <= position <= dataset.shape[0] - 1:
                raise ValueError("The position of the anomaly must be inside "
                                 "the dataset (0 <= pos < dataset.length)")

        match self.anomaly_type:
            case "point":
                offset = random.choice(self.anomaly_params)
                if is_univariate:
                    base_value = dataset[position]
                else:
                    base_value = dataset[position[0], position[1]]
                return offset + base_value

            case "collective":
                match self.collective_type:
                    case "offset":
                        offset = random.choice(self.anomaly_params)
                        return dataset[position:position+self.anomaly_length] + offset

                    case "mean":
                        avg = 0
                        if is_univariate:
                            avg += mean(dataset[position-self.anomaly_params[0]:position])
                        else:
                            pos = list(position)
                            start = pos
                            start[0] -= self.anomaly_params[0]
                            avg += mean(dataset[start:pos])
                        return [avg] * self.anomaly_length

                    case "constant":
                        value = random.choice(self.anomaly_params)
                        return [value] * self.anomaly_length

                    case "function":
                        return self.anomaly_params[0](self.anomaly_length, dataset, position)

    @staticmethod
    def __check_anomaly_params(anomaly_type: str,
                               anomaly_params: list[float],
                               collective_type: str) -> bool:
        """Checks if the parameters are right for that anomaly type

        Parameters
        ----------
        anomaly_type : str
             The type of anomaly.

        anomaly_params : list[float]
            The list of the parameters of the anomaly type.

        collective_type : str
            Type of collective anomaly in case it is collective.

        Returns
        -------
        are_parameters_ok : bool
            True if the parameters for the anomaly are correct, False otherwise.
        """
        match anomaly_type:
            case "point":
                if len(anomaly_params) > 0 and 0 not in anomaly_params:
                    return True

            case "collective":
                match collective_type:
                    case "mean":
                        if len(anomaly_params) == 1 and anomaly_params[0] > 0:
                            return True

                    case "offset":
                        if len(anomaly_params) > 0 and 0 not in anomaly_params:
                            return True

                    case "function":
                        if len(anomaly_params) == 1 and callable(anomaly_params[0]):
                            return True

                    case "constant":
                        if len(anomaly_params) > 0 and 0 not in anomaly_params:
                            return True

        return False
