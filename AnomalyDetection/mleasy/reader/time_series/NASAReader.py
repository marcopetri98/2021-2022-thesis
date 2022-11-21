from __future__ import annotations

import ast
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from reader.time_series import TSReader, rts_config
from utils import print_header, print_step


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

        self.anomalies_path = anomalies_path

        self.__check_parameters()

        self.anomalies_df = pd.read_csv(self.anomalies_path)

    def __iter__(self):
        return NASAIterator(self)

    def __len__(self):
        return 82

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} channels in total")

        channel = self.anomalies_df.iloc[item]["chan_id"]
        return self.read(path=channel, merge_split=True).get_dataframe()

    def read(self, path: str,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             merge_split: bool = False,
             dataset_folder: str = "same-as-labels",
             *args,
             **kwargs) -> NASAReader:
        """
        Parameters
        ----------
        path : str
            The names of the channels to read

        merge_split : bool, default=True
            Whether to merge train and test (concatenate train and test such
            that the new sequence is train -> test). When false, the training
            rows will have prepended "train_" string and the testing rows will
            have prepended "test_" string.

        dataset_folder : str, default="same-as-labels"
            It is the path of the folder containing training and testing splits
            of the dataset. Otherwise, the option "same-as-labels" assumes that
            the dataset folder is the same folder containing the labels.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        elif not isinstance(merge_split, bool):
            raise TypeError("merge_split must be a boolean")
        elif dataset_folder != "same-as-labels" and not os.path.isdir(dataset_folder):
            raise TypeError("dataset_folder must be a valid path to a dir")

        if dataset_folder == "same-as-labels":
            dataset_folder = Path(self.anomalies_path).parent

        if path not in self.anomalies_df["chan_id"].tolist():
            raise ValueError("path must be a valid channel name")
        elif not {"train", "test"}.issubset(os.listdir(dataset_folder)):
            raise ValueError("train and test folders are not present, pass a "
                             "valid dataset folder")

        row_selector = self.anomalies_df["chan_id"] == path

        if verbose:
            print_header("Start reading dataset")
            print_step(f"The selected dataset is {path} of "
                       f"{self.anomalies_df[row_selector]['spacecraft']}")

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
        anomalies = self.anomalies_df.loc[row_selector]["anomaly_sequences"]
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

        if merge_split:
            series = np.concatenate((train_series, test_series))
            self.dataset = pd.DataFrame(series, columns=columns)
        else:
            train_cols = ["train_" + e for e in columns]
            test_cols = ["test_" + e for e in columns]
            data = np.full((train_series.shape[0] + test_series.shape[0], train_series.shape[1] * 2),
                           fill_value=np.nan)
            data[:train_series.shape[0], :train_series.shape[1]] = train_series
            data[train_series.shape[0]:, train_series.shape[1]:] = test_series
            columns = train_cols
            columns.extend(test_cols)
            self.dataset = pd.DataFrame(data, columns=columns)

        self.dataset.insert(0,
                            rts_config["Multivariate"]["index_column"],
                            range(train_series.shape[0] + test_series.shape[0]))
        all_labels = np.concatenate((train_labels, test_labels))
        self.dataset.insert(len(self.dataset.columns),
                            rts_config["Multivariate"]["target_column"],
                            all_labels)

        if verbose:
            print_header("Ended dataset reading")

        return self

    def __check_parameters(self):
        if not isinstance(self.anomalies_path, str):
            raise TypeError("anomalies_path must be a string of a path to a csv"
                            " file")

        if not os.path.isfile(self.anomalies_path):
            raise ValueError("anomalies_path must be a path to a csv file")
