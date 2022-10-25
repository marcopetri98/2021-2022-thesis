from __future__ import annotations

import datetime
import os.path

import pandas as pd

from mleasy.reader.time_series.univariate import UTSAnomalyReader
from mleasy.utils import print_header, print_step


class YahooS5Iterator(object):
    def __init__(self, yahoo_s5):
        super().__init__()
        
        self.index = 0
        self.yahoo_s5 = yahoo_s5
        
    def __next__(self):
        if self.index < len(self.yahoo_s5) - 1:
            self.index += 1
            return self.yahoo_s5[self.index]
        else:
            raise StopIteration()


class YahooS5Reader(UTSAnomalyReader):
    """Data reader for the yahoo webscope S5 anomaly detection dataset.
    
    The class is used to read and access time series contained in the yahoo S5
    dataset for anomaly detection.
    
    Parameters
    ----------
    dataset_location : str
        The location of te dataset in the file system. The location is the
        folder containing the benchmarks and the readme files. If a benchmark is
        not present, an exception will be thrown when trying to load it.
    """
    _ALL_BENCHMARKS = ["A1", "A2", "A3", "A4"]
    _MAX_INT = {
        "A1": 67,
        "A2": 100,
        "A3": 100,
        "A4": 100
    }
    _PREFIX = {
        "A1": "real_",
        "A2": "synthetic_",
        "A3": "A3Benchmark-TS",
        "A4": "A4Benchmark-TS"
    }
    
    def __init__(self, dataset_location: str):
        super().__init__()
        
        self.dataset_location = dataset_location
        
        self.__check_parameters()
        
    def __iter__(self):
        return YahooS5Iterator(self)
        
    def __len__(self):
        return 367
        
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError("use __getitem__ only to iterate over time series")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} series in the dataset")
        
        item = item
        
        if item <= 66:
            benchmark = "A1"
            num = item + 1
        else:
            benchmark = "A" + str(int((item - 67) / 100) + 2)
            num = ((item - 67) % 100) + 1
            
        return self.read(num, benchmark=benchmark).get_dataframe()
        
    def read(self, path: str | int,
             file_format: str = "csv",
             verbose: bool = True,
             benchmark: str = None,
             *args,
             **kwargs) -> YahooS5Reader:
        """
        Parameters
        ----------
        path : str or int
            The absolute path to the csv file of the yahoo dataset, or an
            integer stating which number of time series to load for the
            specified benchmark.
        
        benchmark : ["A1", "A2", "A3", "A4"]
            The benchmark from which the time series must be extracted in case
            `path` is an integer.

        Returns
        -------
        self : YahooS5Reader
            Reference to itself to allow call concatenation.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an int")
        elif benchmark is None:
            raise TypeError(f"benchmark must be one of {self._ALL_BENCHMARKS}")
        
        if isinstance(path, int) and benchmark not in self._ALL_BENCHMARKS:
            raise ValueError(f"benchmark must be one of {self._ALL_BENCHMARKS}")
        elif not 0 < path <= self._MAX_INT[benchmark]:
            raise ValueError(f"for benchmark {benchmark} there are only {self._MAX_INT[benchmark]} series")
        
        if verbose:
            print_header("Dataset reading started")
            print_step("Start reading values")
        
        if isinstance(path, str):
            super().read(path, file_format, False)
        else:
            complete_path = os.path.join(self.dataset_location,
                                         benchmark + "Benchmark",
                                         self._PREFIX[benchmark] + str(path) + ".csv")
            super().read(complete_path, file_format, verbose=False)
            
        if verbose:
            print_step(f"Renaming columns with standard names {[self._TIMESTAMP_COL, self._SERIES_COL]}")
            
        match benchmark:
            case "A1" | "A2":
                self.dataset.rename(columns={
                                        "is_anomaly": self._ANOMALY_COL
                                    },
                                    inplace=True)
        
            case "A3" | "A4":
                self.dataset.rename(columns={
                                        "timestamps": self._TIMESTAMP_COL,
                                        "anomaly": self._ANOMALY_COL
                                    },
                                    inplace=True)
            
        match benchmark:
            case "A2" | "A3" | "A4":
                if verbose:
                    print_step("Converting timestamps into dates")
                    
                dates = [datetime.datetime.fromtimestamp(e) for e in self.dataset[self._TIMESTAMP_COL]]
                self.dataset[self._TIMESTAMP_COL] = pd.to_datetime(dates)
            
        if verbose:
            print_header("Dataset reading ended")
        
        return self
        
    def __check_parameters(self):
        """Check parameters.
        
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            If any parameter has wrong type.
            
        ValueError
            If any parameter has wrong value.
        """
        if not isinstance(self.dataset_location, str):
            raise TypeError("dataset_location must be a string")
        
        if not os.path.isdir(self.dataset_location):
            raise ValueError("dataset_location must be a directory")
