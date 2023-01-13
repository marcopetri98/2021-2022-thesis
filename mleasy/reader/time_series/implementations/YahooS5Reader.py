from __future__ import annotations

import datetime
import os

import pandas as pd

from mleasy.reader.time_series import TSBenchmarkReader, rts_config
from mleasy.utils import print_header, print_step


class YahooS5Iterator(object):
    """An iterator for the YahooS5Reader class.

    The iterator iterates over all the time series present in the Yahoo
    benchmark in lexicographic order.
    """
    def __init__(self, yahoo_s5):
        super().__init__()
        
        self.index = 0
        self.yahoo_s5 = yahoo_s5
        
    def __next__(self):
        if self.index < len(self.yahoo_s5):
            self.index += 1
            return self.yahoo_s5[self.index - 1]
        else:
            raise StopIteration()


class YahooS5Reader(TSBenchmarkReader):
    """Data reader for the yahoo webscope S5 anomaly detection (https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70).
    
    The class is used to read and access time series contained in the yahoo S5
    benchmark for anomaly detection.
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
    
    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)
        
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
            num = item
        else:
            benchmark = "A" + str(int((item - 67) / 100) + 2)
            num = ((item - 67) % 100)
            
        return self.read(num, benchmark=benchmark, verbose=False).get_dataframe()
        
    def read(self, path: str | bytes | os.PathLike | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             benchmark: str = None,
             *args,
             **kwargs) -> YahooS5Reader:
        """
        Parameters
        ----------
        path : str or bytes or PathLike or int
            The path to the csv file of the yahoo dataset, or an integer stating
            which number of time series to load for the specified benchmark.
        
        benchmark : ["A1", "A2", "A3", "A4"]
            The benchmark from which the time series must be extracted in case
            `path` is an integer.

        Returns
        -------
        self : YahooS5Reader
            Reference to itself to allow call concatenation.
        """
        if not isinstance(path, int) and not os.path.isfile(path):
            raise TypeError("path must be a valid path or an int")
        elif benchmark is None:
            raise TypeError(f"benchmark must be one of {self._ALL_BENCHMARKS}")
        
        if isinstance(path, int) and benchmark not in self._ALL_BENCHMARKS:
            raise ValueError(f"benchmark must be one of {self._ALL_BENCHMARKS}")
        elif not 0 <= path < self._MAX_INT[benchmark]:
            raise ValueError(f"for benchmark {benchmark} there are only {self._MAX_INT[benchmark]} series")
        
        if verbose:
            print_header("Dataset reading started")
            print_step("Start reading values")
        
        if isinstance(path, int):
            path = os.path.join(self.benchmark_location,
                                benchmark + "Benchmark",
                                self._PREFIX[benchmark] + str(path + 1) + ".csv")
        super().read(path=path,
                     file_format="csv",
                     pandas_args=pandas_args,
                     verbose=False)
            
        if verbose:
            print_step("Renaming columns with standard names [",
                       rts_config["Univariate"]["index_column"], ", ",
                       rts_config["Univariate"]["value_column"], "]")
            
        match benchmark:
            case "A1" | "A2":
                self.dataset.rename(columns={
                                        "timestamp": rts_config["Univariate"]["index_column"],
                                        "is_anomaly": rts_config["Univariate"]["target_column"],
                                        "value": rts_config["Univariate"]["value_column"]
                                    },
                                    inplace=True)
        
            case "A3" | "A4":
                self.dataset.rename(columns={
                                        "timestamps": rts_config["Univariate"]["index_column"],
                                        "anomaly": rts_config["Univariate"]["target_column"],
                                        "value": rts_config["Univariate"]["value_column"]
                                    },
                                    inplace=True)
            
        match benchmark:
            case "A2" | "A3" | "A4":
                if verbose:
                    print_step("Converting timestamps into dates")
                    
                dates = [datetime.datetime.fromtimestamp(e)
                         for e in self.dataset[rts_config["Univariate"]["target_column"]]]
                self.dataset[rts_config["Univariate"]["target_column"]] = pd.to_datetime(dates)
        
        if verbose:
            print_header("Dataset reading ended")
        
        return self