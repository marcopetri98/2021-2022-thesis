from __future__ import annotations

import os

import pandas as pd

from mleasy.reader.time_series import TSBenchmarkReader, rts_config
from mleasy.utils import print_header, print_step


class GHLIterator(object):
    """An iterator for GHLReader.

    This iterator reads the testing datasets in the order in which they appear
    in the benchmark folder.
    """
    def __init__(self, ghl_reader):
        super().__init__()

        self.index = 0
        self.ghl_reader = ghl_reader

    def __next__(self):
        if self.index < len(self.ghl_reader):
            self.index += 1
            return self.ghl_reader[self.index - 1]
        else:
            raise StopIteration()


class GHLReader(TSBenchmarkReader):
    """Data reader for GHL dataset (https://doi.org/10.48550/arXiv.1612.06676).

    The reader is able to read both testing and training dataset with meaningful
    interfaces.
    """
    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)

        self.__check_parameters()

        self._all_test_sets_paths = list(filter(lambda x: "train" not in x, os.listdir(self.benchmark_location)))
        self._train_set_path = list(filter(lambda x: "train" in x, os.listdir(self.benchmark_location)))[0]

    def __iter__(self):
        return GHLIterator(self)

    def __len__(self):
        return 48

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} testing sets")

        return self.read(path=item).get_dataframe()

    def read(self, path: int | str,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             full_rename: bool = True,
             *args,
             **kwargs) -> GHLReader:
        """
        Parameters
        ----------
        path : int or "train"
            It is the number of the testing set that must be read or a string
            equal to "train" to retrieve the training set.

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        full_rename : bool, default=True
            If `full_rename` is `True` the channels are renamed with integers
            going from 0 to N. Differently, the standard names are kept and the
            standard names are only prepended.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be an integer or a string")
        elif isinstance(path, str) and path != "train":
            raise TypeError("path can only be \"train\" if it is a string")
        elif not isinstance(full_rename, bool):
            raise TypeError("full_rename must be boolean")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"there are only {len(self)} testing sets")

        if verbose:
            print_header("Start to read dataset")
            if path == "train":
                print_step("Reading GHL training dataset")
            else:
                print_step(f"Reading GHL testing set {path}")

        if isinstance(path, int):
            file_path = os.path.join(self.benchmark_location, self._all_test_sets_paths[path])
        else:
            file_path = os.path.join(self.benchmark_location, self._train_set_path)

        if verbose:
            print_step("Reading file and ordering columns")

        # read file and reorder columns
        dataset = pd.read_csv(file_path)
        ordered_cols = [e for e in dataset.columns if e not in ["DANGER", "FAULT", "ATTACK"]]
        ordered_cols.extend(["DANGER", "FAULT", "ATTACK"])
        dataset = dataset[ordered_cols]

        if verbose:
            print_step("Renaming columns with standard names")

        # build columns name mappings
        channels = {e: f"channel_{e if not full_rename else idx}"
                    for idx, e in enumerate(dataset.columns[1:-3])
                    if e not in ["DANGER", "FAULT", "ATTACK"]}
        classes = {e: f"class_{e if not full_rename else idx}"
                   for idx, e in enumerate(["DANGER", "FAULT", "ATTACK"])}

        # rename columns
        dataset.rename(columns={"Time": rts_config["Multivariate"]["index_column"]},
                       inplace=True)
        dataset.rename(columns=channels, inplace=True)
        dataset.rename(columns=classes, inplace=True)

        self.dataset = dataset.copy()

        if verbose:
            print_header("Ended dataset reading")

        return self

    def __check_parameters(self):
        if len(os.listdir(self.benchmark_location)) != len(self) + 1:
            raise ValueError("benchmark_location must contain all the 48 tests "
                             "and the training set")
