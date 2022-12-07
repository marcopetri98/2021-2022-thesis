from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd

from mleasy.reader.time_series import TSBenchmarkReader, rts_config
from mleasy.utils.printing import print_header, print_step
from mleasy.utils import load_py_json


class NABIterator(object):
    """An iterator for the NAB benchmark.

    The iterator reads from the first to the last ordered datasets' folders and
    csv files. It also reads the datasets without anomalies.
    """
    def __init__(self, nab_reader):
        super().__init__()

        self.index = 0
        self.nab_reader = nab_reader

    def __next__(self):
        if self.index < len(self.nab_reader):
            self.index += 1
            return self.nab_reader[self.index - 1]
        else:
            raise StopIteration()


class NABReader(TSBenchmarkReader):
    """A reader of NAB time series datasets.

    The reader reads the time series and adds the target column class for the
    time series defined as it is defined by NAB (windows such that the sum of
    the windows' length is 10% of data around each label point).
    """
    
    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)

        self._datasets_paths = []
        self._datasets_names = []

        data_path = os.path.join(self.benchmark_location, "data")
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file != "README.md":
                    self._datasets_names.append(file.split(".")[0])
                    self._datasets_paths.append(os.path.join(root, file))

        labels_path = os.path.join(self.benchmark_location, "labels")
        windows_path = os.path.normpath(os.path.join(labels_path, "combined_windows.json"))
        with open(windows_path) as file:
            self._combined_windows = json.load(file)

        self._combined_windows = {key.split("/")[1].split(".")[0]: self._combined_windows[key]
                                  for key in self._combined_windows}

    def __iter__(self):
        return NABIterator(self)

    def __len__(self):
        return 58

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"item must be less than {len(self)}")

        return self.read(path=item).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> NABReader:
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an int")
        elif isinstance(path, str) and path not in self._datasets_names:
            raise ValueError(f"path must be one of {self._datasets_names}")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"there are only {len(self)} datasets in NAB")
        
        if verbose:
            print_header("Start reading dataset")
        
        # get the dataset path
        if isinstance(path, str):
            dataset_path = self._datasets_paths[self._datasets_names.index(path)]
            dataset_name = path
        else:
            dataset_path = self._datasets_paths[path]
            dataset_name = self._datasets_names[path]

        if verbose:
            print_step(f"Loading series from file path {os.path.normpath(dataset_path)}")

        # load dataset
        dataset = pd.read_csv(dataset_path)

        if verbose:
            print_step("Start to read and build labels")
        
        # build target class vector
        target = np.zeros(dataset.shape[0])
        if len(self._combined_windows[dataset_name]) != 0:
            for window in self._combined_windows[dataset_name]:
                start_idx = dataset["timestamp"].tolist().index(window[0].split(".")[0])
                end_idx = dataset["timestamp"].tolist().index(window[1].split(".")[0])
                target[start_idx:end_idx + 1] = 1

        if verbose:
            print_step("Renaming columns with standard names")
        
        # give to the columns standard names
        dataset.rename(columns={"timestamp": rts_config["Univariate"]["index_column"]},
                       inplace=True)
        dataset.rename(columns={"value": rts_config["Univariate"]["value_column"]},
                       inplace=True)
        dataset.insert(len(dataset.columns),
                       rts_config["Univariate"]["target_column"],
                       target)

        self.dataset = dataset.copy()

        if verbose:
            print_header("Ended dataset reading")
        
        return self

    def __check_parameters(self):
        if len(os.listdir(self.benchmark_location)) != 2:
            raise ValueError("benchmark_location must contain only data and "
                             "labels folders")

        labels_path = os.path.join(self.benchmark_location, "labels")
        data_path = os.path.join(self.benchmark_location, "data")

        if "combined_windows.json" not in os.listdir(labels_path):
            raise ValueError("labels folder does not contain combined_windows "
                             "file")

        num_dirs = 0
        num_files = 0
        for root, dirs, files in os.walk(data_path):
            num_dirs += len(dirs)
            num_files += len(files)

        if num_dirs != 7 or num_files != len(self) + 1:
            raise ValueError("data folder should contain the 7 NAB folders and "
                             f"all the {len(self)} files and the readme")
