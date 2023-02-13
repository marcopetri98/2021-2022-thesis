from __future__ import annotations

import os

from .. import TSBenchmarkReader, rts_config
from ....utils import print_header, print_step


class MGABReaderIterator(object):
    """An iterator for the MGABReader class.

    The iterator iterates over time series in ascending order.
    """
    def __init__(self, mgab_reader):
        super().__init__()

        self.index = 0
        self.mgab_reader = mgab_reader

    def __next__(self):
        if self.index < len(self.mgab_reader):
            self.index += 1
            return self.mgab_reader[self.index - 1]
        else:
            raise StopIteration()


class MGABReader(TSBenchmarkReader):
    """Data reader for MGAB anomaly benchmark (https://doi.org/10.5281/zenodo.3760086).

    This reader is used to read the datasets contained in the MGAB benchmark.
    """

    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

    def __iter__(self):
        return MGABReaderIterator(self)

    def __len__(self):
        return 10

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError("use __getitem__ only to iterate over time series")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} series in the dataset")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: str | bytes | os.PathLike | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> MGABReader:
        """
        Parameters
        ----------
        path : str or bytes or PathLike or int
            The path to the csv containing the time series or the integer
            representing which time series to load from the dataset location.

        Returns
        -------
        self : MGABReader
            An instance to itself to allow call chaining.
        """
        if not isinstance(path, int) and not os.path.isfile(path):
            raise TypeError("path must be a valid path or an int")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"path must be between 0 and {len(self)}")

        if verbose:
            print_header("Dataset reading started")
            print_step("Start reading values")

        if isinstance(path, int):
            path = self._benchmark_path / (str(path + 1) + ".csv")

        super().read(path=path,
                     file_format=file_format,
                     pandas_args=pandas_args,
                     verbose=False)

        if verbose:
            print_step("Renaming columns with standard names [",
                       rts_config["Univariate"]["index_column"], ", ",
                       rts_config["Univariate"]["value_column"], "]")

        self._dataset.rename(columns={
                                "Unnamed: 0": rts_config["Univariate"]["index_column"],
                                "value": rts_config["Univariate"]["value_column"],
                                "is_anomaly": rts_config["Univariate"]["target_column"]
                            },
                            inplace=True)

        if verbose:
            print_header("Dataset reading ended")

        return self
