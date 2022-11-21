from __future__ import annotations

import os

from mleasy.reader.time_series import TSBenchmarkReader
from mleasy.utils import print_header, print_step


class MGABReader(TSBenchmarkReader):
    _ANOMALY_COL = "target"
    _SERIES_COL = "value"
    _TIMESTAMP_COL = "timestamp"

    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> MGABReader:
        """
        Parameters
        ----------
        path : str or int
            The path to the csv containing the time series or the integer
            representing which time series to load from the dataset location.

        Returns
        -------
        self : MGABReader
            An instance to itself to allow call chaining.
        """
        if not isinstance(path, str) or not isinstance(path, int):
            raise TypeError("path must be a string or an int")

        if verbose:
            print_header("Start to read MGAB dataset")

        if isinstance(path, int):
            path = os.path.join(self.benchmark_location, str(path), ".csv")

        if verbose:
            print_step(f"Reading file {path}")

        super().read(path=path, file_format=file_format, verbose=False)

        print(self.dataset)

        return self
