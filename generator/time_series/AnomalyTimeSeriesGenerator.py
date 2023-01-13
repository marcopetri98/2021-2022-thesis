from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Union
import random
import datetime

import numpy as np
import pandas as pd

from generator.printing.DecoratedPrinter import DecoratedPrinter
from generator.time_series.Anomaly import Anomaly
from generator.time_series.AnomalyDistribution import AnomalyDistribution
from generator.time_series.Seasonality import Seasonality
from generator.time_series.TimeSeriesGenerator import TimeSeriesGenerator
from generator.time_series.Trend import Trend


@dataclass(frozen=True)
class AnomalyTimeSeriesPrints(object):
    ERROR_IMPL_ANOMALY = "Anomaly type not implemented"
    ERROR_TOO_MANY_ANOMALIES = "Anomalies must be in the interval (0, 0.2]"
    ERROR_TOO_SHORT = "A collective anomaly must have at least 2 points"
    ERROR_WRONG_PARAMS = "The parameters are wrong for the specified anomaly"

    GENERATE = ["Anomaly detection dataset generation started",
                "Anomaly detection dataset generation ended",
                "Generation of the time series started",
                "Generate anomalies' position",
                "Add the anomalies on the dataset",
                "Creating the ground truth to populate",
                "Saving the dataframe of the dataset"]


class AnomalyTimeSeriesGenerator(TimeSeriesGenerator):
    """Class defining anomaly time series dataset generation

    Attributes
    ----------
    anomalies
        A string representing the anomalies present in the dataset.

    anomalies_perc
        Percentage of anomalies in the dataset.
    """
    ALLOWED_ANOMALIES = ["point",
                         "collective"]

    def __init__(self, supervised: bool,
                 labels: list[str]):
        super().__init__(supervised, labels)
        self.anomalies = None
        self.anomalies_perc = None

    def generate(self, num_points: int,
                 dimensions: int,
                 stochastic_process: str,
                 process_params: list[float],
                 noise: list[str],
                 custom_noise: list[Callable[[], float]] = None,
                 verbose: bool = True,
                 sample_freq_seconds: float = 1.0,
                 trend: list[bool] = None,
                 seasonality: list[bool] = None,
                 trend_func: list[Trend] = None,
                 seasonality_func: list[Seasonality] = None,
                 columns_names: list[str] = None,
                 start_timestamp: float = datetime.datetime.now().timestamp(),
                 anomalies: Union[list[Anomaly], list[list[Anomaly]]] = None,
                 anomaly_dist: AnomalyDistribution = AnomalyDistribution("uniform"),
                 anomalies_perc: float = 0.01,
                 *args, **kwargs) -> AnomalyTimeSeriesGenerator:
        """Generate an anomaly detection time series dataset.

        Parameters
        ----------
        anomalies : list[Anomaly] or list[list[Anomaly]], default=None
            All the possible anomalies we can encounter in the dataset being
            generated. They can even be of different types one from another.

        anomaly_dist : AnomalyDistribution, default=AnomalyDistribution("uniform")
            The distribution of anomalies on the time series.

        anomalies_perc : float, default=0.01
            Percentage of anomalies in the dataset.

        Notes
        -----
        The parameter anomalies must be given to the class to be able to create
        an anomaly detection dataset.
        """
        if anomalies is None:
            raise ValueError("anomalies parameter must be given")
        elif anomalies_perc <= 0 or anomalies_perc >= 1:
            raise ValueError(AnomalyTimeSeriesPrints.ERROR_TOO_MANY_ANOMALIES)
        elif isinstance(anomalies[0], list):
            raise NotImplementedError("Only univariate implemented")

        if verbose:
            DecoratedPrinter.print_heading(AnomalyTimeSeriesPrints.GENERATE[0])
            DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[2])

        super().generate(num_points,
                         dimensions,
                         stochastic_process,
                         process_params,
                         noise,
                         custom_noise,
                         False,
                         sample_freq_seconds,
                         trend,
                         seasonality,
                         trend_func,
                         seasonality_func,
                         columns_names,
                         start_timestamp)

        if columns_names is not None and self.supervised:
            columns_names = columns_names + ["target_"+c for c in columns_names]

        num_points = self.dataset.shape[0]
        num_anomalies = int(num_points * anomalies_perc)

        if verbose:
            DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[3])
        # TODO: implement also multivariate
        # Generate indexes at which anomalies are found
        indexes = anomaly_dist.generate_anomaly_positions(num_points,
                                                          num_anomalies)

        # Generate the anomalies
        anomalies_generated = 0

        if self.supervised:
            if verbose:
                DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[5])
            self.ground_truth = np.ndarray(self.dataset.shape, dtype=np.intc)
            self.ground_truth.fill(0)

        if verbose:
            DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[4])

        for idx in indexes:
            if anomalies_generated >= num_anomalies:
                break

            anomaly = random.choice(anomalies)
            generated = anomaly.compute_anomaly(self.dataset, idx)
            if isinstance(generated, list):
                # In case collective is generated at the end, we do clipping
                clipping = len(self.dataset[idx:idx + len(generated)])
                self.dataset[idx:idx + len(generated)] = generated[0:clipping]
                if self.supervised:
                    self.ground_truth[idx:idx + len(generated)] = 1
            else:
                self.dataset[idx] = generated
                if self.supervised:
                    self.ground_truth[idx] = 1

        if verbose:
            DecoratedPrinter.print_step(AnomalyTimeSeriesPrints.GENERATE[6])

        self.__save_anomaly(num_points,
                            dimensions,
                            columns_names)
        if verbose:
            DecoratedPrinter.print_heading(AnomalyTimeSeriesPrints.GENERATE[1])

        return self

    def __save_anomaly(self, num_points: int,
                       dimensions: int,
                       columns_names: list[str]) -> None:
        """Saves the anomaly dataframe version of the dataset

        Parameters
        ----------
        num_points : int
            Number of points in the dataset.

        dimensions : int
            Number of dimensions of the dataset. CURRENTLY, ONLY SUPPORTS 1.

        columns_names : list[str]
            List of the column names to use for the time series. If given,
            len(columns_names) must be identical to dim.

        Returns
        -------

        """
        timestamps = [pd.Timestamp(x, unit="s") for x in self.dataset_timestamps]
        index = pd.DatetimeIndex(timestamps)
        index.name = "timestamp"

        if not self.supervised:
            if columns_names is None:
                self.dataset_frame = pd.DataFrame(self.dataset,
                                                  index,
                                                  dtype=np.double)
            else:
                self.dataset_frame = pd.DataFrame(self.dataset,
                                                  index,
                                                  columns_names,
                                                  dtype=np.double)
        else:
            new_shape = (num_points, dimensions * 2)
            complete_ds: np.ndarray = np.ndarray(new_shape, dtype=object)
            complete_ds[:, 0:dimensions] = np.reshape(self.dataset, (num_points, 1))
            complete_ds[:, dimensions:2 * dimensions] = np.reshape(self.ground_truth, (num_points, 1))
            self.dataset = complete_ds

            if columns_names is None:
                self.dataset_frame = pd.DataFrame(complete_ds,
                                                  index,
                                                  dtype=object)
            else:
                self.dataset_frame = pd.DataFrame(complete_ds,
                                                  index,
                                                  columns_names,
                                                  dtype=object)
