from __future__ import annotations

import ast
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from .. import TSReader, rts_config
from ....utils import print_header, print_step


class NASAIterator(object):
    """An iterator for NASAReader.

    The iterator iterates over all the channels of the two time series in the
    same order of the labeled anomalies file.
    """
    def __init__(self, nasa_reader):
        super().__init__()

        self.index = 0
        self.nasa_reader = nasa_reader

    def __next__(self):
        if self.index < len(self.nasa_reader):
            self.index += 1
            return self.nasa_reader[self.index - 1]
        else:
            raise StopIteration()


class NASAReader(TSReader):
    """Data reader for NASA MSL and NASA SMAP datasets (https://doi.org/10.1145/3219819.3219845).

    The reader is set up such that NASA files pre-split in training and testing
    are read as is. Eventually, with a flag is possible to decide to merge
    training and testing data to build a single unique dataframe containing all
    data such that new and different split can be performed.
    """
    def __init__(self, anomalies_path: str):
        super().__init__()

        self._anomalies_path = anomalies_path

        self.__check_parameters()

        self._anomalies_df = pd.read_csv(self._anomalies_path)

    def __iter__(self):
        return NASAIterator(self)

    def __len__(self):
        return 82

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} channels in total")

        return self.read(path=item, merge_split=True, verbose=False).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             dataset_folder: str = "same-as-labels",
             *args,
             **kwargs) -> NASAReader:
        """
        Parameters
        ----------
        path : str or int
            The names of the channels to read (e.g., "A-1" is a valid value
            for path), or an integer stating which time series to load from
            the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        dataset_folder : str, default="same-as-labels"
            It is the path of the folder containing training and testing splits
            of the dataset. Otherwise, the option "same-as-labels" assumes that
            the dataset folder is the same folder containing the labels.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an integer")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise IndexError(f"path is {path} and NASA has {len(self)} series")
        elif dataset_folder != "same-as-labels" and not os.path.isdir(dataset_folder):
            raise TypeError("dataset_folder must be a valid path to a dir")

        if isinstance(path, int):
            path = self._anomalies_df.iloc[path]["chan_id"]

        if dataset_folder == "same-as-labels":
            dataset_folder = Path(self._anomalies_path).parent

        if path not in self._anomalies_df["chan_id"].tolist():
            raise ValueError("path must be a valid channel name")
        elif not {"train", "test"}.issubset(os.listdir(dataset_folder)):
            raise ValueError("train and test folders are not present, pass a "
                             "valid dataset folder")

        row_selector = self._anomalies_df["chan_id"] == path

        if verbose:
            print_header("Start reading dataset")
            print_step(f"The selected dataset is {path} of "
                       f"{self._anomalies_df[row_selector]['spacecraft']}")

        # if the user specified one of the channels build the path
        train_path = os.path.join(dataset_folder, "train", path + ".npy")
        test_path = os.path.join(dataset_folder, "test", path + ".npy")

        if verbose:
            print_step("Reading training and testing data")

        train_series = np.load(train_path)
        test_series = np.load(test_path)

        if verbose:
            print_step("Build target class vector")

        train_labels = np.zeros(train_series.shape[0])
        test_labels = np.zeros(test_series.shape[0])
        anomalies = self._anomalies_df.loc[row_selector]["anomaly_sequences"]
        anomalies = ast.literal_eval(anomalies.iloc[0])
        for sequence in anomalies:
            test_labels[sequence[0]:sequence[1] + 1] = 1

        if verbose:
            print_step("Naming columns with default names")

        columns = [str(e) for e in range(train_series.shape[1])]
        columns[0] = "telemetry"
        columns = [rts_config["Multivariate"]["channel_column"] + "_" + e
                   for e in columns]

        if verbose:
            print_step("Building the dataframe")

        series = np.concatenate((train_series, test_series))
        targets = np.concatenate((train_labels, test_labels))
        timestamp = np.arange(series.shape[0])
        is_training = np.zeros(series.shape[0])
        is_training[:train_series.shape[0]] = 1
        all_columns = [rts_config["Multivariate"]["index_column"]]
        all_columns.extend(columns)
        all_columns.append(rts_config["Multivariate"]["target_column"])
        all_columns.append(rts_config["Multivariate"]["is_training"])
        all_data = np.concatenate((timestamp.reshape(-1, 1),
                                   series,
                                   targets.reshape(-1, 1),
                                   is_training.reshape(-1, 1)),
                                  axis=1)
        self._dataset = pd.DataFrame(all_data, columns=all_columns)

        if verbose:
            print_header("Ended dataset reading")

        return self

    def __check_parameters(self):
        if not isinstance(self._anomalies_path, str):
            raise TypeError("anomalies_path must be a string of a path to a csv"
                            " file")

        if not os.path.isfile(self._anomalies_path):
            raise ValueError("anomalies_path must be a path to a csv file")
