from typing import Tuple

import numpy as np


class AnomalyDistribution(object):
    """A class representing an anomaly distribution.

    Attributes
    ----------
    anomaly_dist : str
        String representing the anomaly distribution over the dataset.

    dist_params : list[float]
        Parameters of the distribution.
    """
    ALLOWED_ANOMALY_DIST = ["uniform",
                            "triangular",
                            "beta"]

    def __init__(self, anomaly_dist: str,
                 dist_params: list[float] = None):
        super().__init__()
        if dist_params is None:
            dist_params = []

        self.anomaly_dist, self.params = self.set_distribution(anomaly_dist,
                                                               dist_params)

    def set_distribution(self, dist: str,
                         dist_params: list[float]) -> Tuple[str, list[float]]:
        """Changes the anomaly distribution.

        Parameters
        ----------
        dist : str
            The anomaly distribution over the dataset.

        dist_params : list[float]
            Parameters of the distribution.
        """
        if dist not in self.ALLOWED_ANOMALY_DIST:
            raise ValueError("Anomaly distribution invalid")
        elif not self.__check_dist_params(dist, dist_params):
            raise ValueError("Distribution parameters are wrong")

        return dist, dist_params.copy()

    def generate_anomaly_positions(self, total_points: int,
                                   num_anomalies: int | Tuple[int]) -> list[int] | list[list[int]]:
        """Generate indexes of anomalies for the dataset sampling a distribution.

        Parameters
        ----------
        total_points : int
            Number of total points in the dataset.

        num_anomalies : int or tuple[int]
            Number of anomalies position to generate.

        Notes
        -----
        Some distributions generate points between 0 and 1. For those
        distributions, remember that the generated values are projected onto
        the dataset dimensions, i.e., if 0<=x<=1 is the generated number, then,
        the generated index will be x*(total_points-1).

        The distributions whose range can be defined by the user directly define
        the index that is being generated.
        """
        if total_points <= num_anomalies:
            raise ValueError("Anomalies must be less than points")
        elif total_points <= 1:
            raise ValueError("There must enough points for anomalies and not")

        if isinstance(num_anomalies, int):
            return self.__generate_indexes(total_points, num_anomalies)
        else:
            return [self.__generate_indexes(total_points, x) for x in num_anomalies]

    def __generate_indexes(self, total_points: int,
                           num_anomalies: int) -> list[int]:
        """Generates a list of indexes for the anomalies to generate.

        Parameters
        ----------
        total_points : int
            Number of total points in the dataset.

        num_anomalies : int or tuple[int]
            Number of anomalies position to generate.
        """
        indexes = []

        match self.anomaly_dist:
            case "uniform":
                if len(self.params) == 0:
                    indexes = np.random.uniform(0,
                                                total_points - 1,
                                                num_anomalies)
                else:
                    indexes = np.random.uniform(self.params[0],
                                                self.params[1],
                                                num_anomalies)

            case "triangular":
                indexes = np.random.triangular(self.params[0],
                                               self.params[1],
                                               self.params[2],
                                               num_anomalies)

            case "beta":
                indexes = np.random.beta(self.params[0],
                                         self.params[1],
                                         num_anomalies)
                indexes = [x * (total_points - 1) for x in indexes]

        return [int(x) for x in indexes]

    @staticmethod
    def __check_dist_params(dist: str,
                            dist_params: list[float]) -> bool:
        """Check that the given parameters are right

        Parameters
        ----------
        dist : str
            The name of the distribution.

        dist_params : list[float]
            The parameters of the distribution.

        Returns
        -------
        are_parameters_correct : bool
            True when the passed parameters are correct for the chosen
            distribution, False otherwise.
        """

        match dist:
            case "uniform":
                if len(dist_params) == 2 and dist_params[0] < dist_params[1]:
                    return True
                else:
                    if len(dist_params) != 0:
                        return False

            case "triangular":
                if (len(dist_params) == 3 and
                        dist_params[0] < dist_params[2] and
                        dist_params[0] <= dist_params[1] <= dist_params[2]):
                    return True

            case "beta":
                if (len(dist_params) == 2 and
                        dist_params[0] > 0 and
                        dist_params[1] > 0):
                    return True

        return False
