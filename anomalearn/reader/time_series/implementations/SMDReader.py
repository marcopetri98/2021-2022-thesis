from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .. import TSBenchmarkReader, rts_config
from ....utils import print_header, print_step


class SMDIterator(object):
    """An iterator for SMDReader.

    The iterator reads the datasets from the first machine till the last machine
    in lexicographic order.
    """
    def __init__(self, smd_reader):
        super().__init__()

        self.index = 0
        self.smd_reader = smd_reader

    def __next__(self):
        if self.index < len(self.smd_reader):
            self.index += 1
            return self.smd_reader[self.index - 1]
        else:
            raise StopIteration()


class SMDReader(TSBenchmarkReader):
    """Data reader for SMD dataset (https://doi.org/10.1145/3292500.3330672).

    The reader reads the txt files in the SMD benchmark folder and translates
    them into the default format for time series.
    """
    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)

        self._interpretation = os.path.join(self.benchmark_location, "interpretation_label")
        self._test_set = os.path.join(self.benchmark_location, "test")
        self._test_gt = os.path.join(self.benchmark_location, "test_label")
        self._train_set = os.path.join(self.benchmark_location, "train")
        
        self._machines = [e.split(".txt")[0]
                          for e in os.listdir(self._train_set)
                          if os.path.isfile(os.path.join(self._train_set, e))]
        self._machines.sort(key=lambda elem: int(elem.split("-")[1])*10 + int(elem.split("-")[2]))

        self.__check_parameters()

    def __iter__(self):
        return SMDIterator(self)

    def __len__(self):
        return 28

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} machines")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> SMDReader:
        """
        Parameters
        ----------
        path : str or int
            It is the name of the machine that you want to read (e.g.,
            machine-1-1"), or an integer stating which time series to load from
            the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a machine name or an index")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise IndexError(f"path is {path} and SMD has {len(self)} series")
        elif isinstance(path, str) and path not in self._machines:
            raise ValueError(f"path must be one of {self._machines}")

        if isinstance(path, int):
            path = self._machines[path]

        if verbose:
            print_header("Started dataset reading")
            print_step(f"Reading machine {path}")

        # read training dataset and testing
        training_set = pd.read_csv(os.path.join(self._train_set, path + ".txt"),
                                   header=None)
        testing_set = pd.read_csv(os.path.join(self._test_set, path + ".txt"),
                                  header=None)
        test_labels = pd.read_csv(os.path.join(self._test_gt, path + ".txt"),
                                  header=None)[0].values

        if verbose:
            print_step("Renaming columns with standard names")

        # retrieve number of columns and mapping to new columns' names
        dataset_header = [f"{rts_config['Multivariate']['channel_column']}_{e}"
                          for e in training_set.columns]
        columns_mapping = {training_set.columns[i]: dataset_header[i]
                           for i in range(len(dataset_header))}

        # rename columns with standard names
        training_set.rename(columns=columns_mapping, inplace=True)
        testing_set.rename(columns=columns_mapping, inplace=True)

        if verbose:
            print_step("Building labels and is_training column")

        # build overall labels and training column
        labels = np.zeros(training_set.shape[0] + testing_set.shape[0])
        labels[training_set.shape[0]:] = test_labels
        is_training = np.zeros(training_set.shape[0] + testing_set.shape[0])
        is_training[:training_set.shape[0]] = 1
        interpretation = [None] * labels.shape[0]

        if verbose:
            print_step("Reading anomalies' interpretation")

        # reading the interpretation file
        with open(os.path.join(self._interpretation, path + ".txt"), "r") as f:
            for line in f:
                interval, channels = line.split(":")
                start, end = interval.split("-")
                elements = channels.split("\n")[0].split(",")
                elements = [int(e) for e in elements]
                for i in range(int(start), int(end) + 1, 1):
                    interpretation[i] = elements

        if verbose:
            print_step("Building the overall dataset")

        # build the overall dataset
        self._dataset = pd.concat((training_set, testing_set))
        self._dataset.set_index(np.arange(self._dataset.shape[0]), inplace=True)
        self._dataset.insert(0,
                             rts_config["Multivariate"]["index_column"],
                             np.arange(self._dataset.shape[0]))
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Multivariate"]["target_column"],
                             labels)
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Multivariate"]["is_training"],
                             is_training)
        self._dataset.insert(len(self._dataset.columns),
                            "interpretation",
                             interpretation)

        if verbose:
            print_header("Dataset reading ended")

        return self

    def __check_parameters(self):
        if not os.path.isdir(self._interpretation):
            raise ValueError("benchmark_location must contain a folder named "
                             "interpretation_label")
        elif not os.path.isdir(self._test_gt):
            raise ValueError("benchmark_location must contain a folder named "
                             "test_label")
        elif not os.path.isdir(self._test_set):
            raise ValueError("benchmark_location must contain a folder named "
                             "test")
        elif not os.path.isdir(self._train_set):
            raise ValueError("benchmark_location must contain a folder named "
                             "train")
