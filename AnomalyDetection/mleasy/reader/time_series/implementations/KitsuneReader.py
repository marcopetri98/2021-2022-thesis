from __future__ import annotations
import os

import numpy as np
import pandas as pd

from mleasy.reader.time_series import TSBenchmarkReader, rts_config
from mleasy.utils import print_step, print_header


class KitsuneIterator(object):
    """An iterator for KitsuneReader.
    
    The iterator reads the datasets in lexicographic order.
    """
    def __init__(self, kitsune_reader):
        super().__init__()
        
        self.index = 0
        self.kitsune_reader = kitsune_reader
        
    def __next__(self):
        if self.index < len(self.kitsune_reader):
            self.index += 1
            return self.kitsune_reader[self.index - 1]
        else:
            raise StopIteration()


class KitsuneReader(TSBenchmarkReader):
    """A reader for the Kitsune benchmark.
    
    The reader reads the csv files with the raw data and the labels. The reader
    combines them in the standard format.
    """
    _DATASETS = ["active_wiretap", "arp_mitm", "fuzzing", "mirai", "os_scan",
                 "ssdp_flood", "ssl_renegotiation", "syn_dos", "video_injection"]
    
    def __init__(self, benchmark_location: str):
        super().__init__(benchmark_location=benchmark_location)
        
        self.__check_parameters()
        
    def __iter__(self):
        return KitsuneIterator(self)
        
    def __len__(self):
        return 9
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer to be able to index")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} datasets")
        
        return self.read(path=item, verbose=False).get_dataframe()
    
    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             *args,
             **kwargs) -> KitsuneReader:
        """
        Parameters
        ----------
        path : str or int
            It is the name of the dataset that the method must read or the
            number of the dataset that will be retrieved by the iterator.

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an int")
        elif not isinstance(path, int) and path not in self._DATASETS:
            raise ValueError(f"path must be one of {self._DATASETS}")
        elif not isinstance(path, str) and not 0 <= path < len(self):
            raise ValueError(f"there are only {len(self)} datasets")
        
        dataset_name = path if isinstance(path, str) else self._DATASETS[path]
        dataset_path = os.path.normpath(os.path.join(self.benchmark_location, dataset_name))
        
        if verbose:
            print_header("Started dataset reading")
            print_step(f"Reading {dataset_name} dataset located in folder {dataset_path}")
            print_step("Reading raw data and raw labels")
            
        print(os.path.join(dataset_path, dataset_name + "_dataset.csv"))
        # read raw data and labels
        dataset = pd.read_csv(os.path.join(dataset_path, dataset_name + "_dataset.csv"),
                              header=None,
                              dtype=float)
        if dataset_name == "mirai":
            raw_labels = pd.read_csv(os.path.join(dataset_path, dataset_name + "_labels.csv"),
                                     header=None,
                                     dtype=int)
            labels = raw_labels[raw_labels.columns[0]].values
        else:
            raw_labels = pd.read_csv(os.path.join(dataset_path, dataset_name + "_labels.csv"))
            labels = raw_labels["x"].values
            
        
        if verbose:
            print_step("Renaming columns with standard names")
            
        # rename columns with standard names
        dataset_header = [f"{rts_config['Multivariate']['channel_column']}_{e}"
                          for e in dataset.columns]
        columns_mapping = {dataset.columns[i]: dataset_header[i]
                           for i in range(len(dataset_header))}

        # rename columns with standard names
        dataset.rename(columns=columns_mapping, inplace=True)
        
        if verbose:
            print_step("Building the overall dataset")
            
        dataset.set_index(np.arange(dataset.shape[0]), inplace=True)
        dataset.insert(0,
                       rts_config["Multivariate"]["index_column"],
                       np.arange(dataset.shape[0]))
        dataset.insert(len(dataset.columns),
                       rts_config["Multivariate"]["target_column"],
                       labels)
        self.dataset = dataset.copy()
        
        if verbose:
            print_header("Dataset reading ended")
        
        return self
        
    def __check_parameters(self):
        contents = [os.path.join(self.benchmark_location, e)
                    for e in os.listdir(self.benchmark_location)]
        dirs = list(filter(os.path.isdir, contents))
        
        if len(dirs) != 9:
            raise ValueError("benchmark_location must contain the 9 datasets")
        
        for dir_path in dirs:
            if len(os.listdir(dir_path)) != 3:
                raise ValueError("each dataset directory must have exactly 3 "
                                 "files")
