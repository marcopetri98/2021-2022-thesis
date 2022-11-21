from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from mleasy.reader import MissingStrategy
from mleasy.reader.time_series import TSReader
from mleasy.utils.printing import print_header, print_step


class ODINTSReader(TSReader):
    """A reader for ODIN TS annotated datasets.
    
    Parameters
    ----------
    anomalies_path : str
        It is the file path in which the anomalies are stored as csv.
    
    timestamp_col : str
        It is the column with the timestamps of data.
    
    univariate_col : str
        It is the column on which the dataset has been labelled.
    
    start_col : str, default="start_date"
        It is the column of the anomalies file stating the start of an anomaly
        window.
    
    end_col : str, default="end_date"
        It is the column of the anomalies file stating the end of an anomaly
        window.
    """
    _ANOMALY_COL = "target"
    _SERIES_COL = "value"
    _TIMESTAMP_COL = "timestamp"
    _DAY_COL = "day_of_the_week"
    _ANOMALY_TYPE = "anomaly_type"
    
    def __init__(self, anomalies_path: str,
                 timestamp_col: str,
                 univariate_col: str,
                 start_col: str = "start_date",
                 end_col: str = "end_date",
                 anomaly_type_col: str = "anomaly_type"):
        super().__init__()
        
        self.anomalies_path = anomalies_path
        self.timestamp_col = timestamp_col
        self.univariate_col = univariate_col
        self.start_col = start_col
        self.end_col = end_col
        self.anomaly_type_col = anomaly_type_col
        
        self._unmodified_dataset: pd.DataFrame = None
        
    def read(self, path: str,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             verbose: bool = True,
             resample: bool = False,
             resampling_granularity: str = "1min",
             missing_strategy: MissingStrategy = MissingStrategy.DROP,
             missing_fixed_value: float = 0.0,
             *args,
             **kwargs) -> ODINTSReader:
        # TODO: implement interpolation imputation
        if missing_strategy not in [MissingStrategy.NOTHING, MissingStrategy.DROP, MissingStrategy.FIXED_VALUE]:
            raise NotImplementedError("Imputation still not implemented")

        if verbose:
            print_header("Start reading dataset")
            print_step("Start reading the dataset values")
        
        super().read(path, file_format, verbose=False)
        
        if verbose:
            print_step("Ended dataset values reading")
        
        self._unmodified_dataset = self.dataset.copy()
        
        dataset_cp = self._add_information(verbose=verbose)
        
        if verbose:
            print_step("Renaming columns with standard names {}".format([self._TIMESTAMP_COL,
                                                                         self._SERIES_COL]))
        
        # add anomaly labels to original dataset and drop useless columns
        self.dataset.insert(len(self.dataset.columns),
                            self._ANOMALY_COL,
                            dataset_cp[self._ANOMALY_COL].values)
        self.dataset.rename(columns={
                                self.timestamp_col: self._TIMESTAMP_COL,
                                self.univariate_col: self._SERIES_COL
                            },
                            inplace=True)
        self.dataset.drop(columns=self.dataset.columns.difference([self._TIMESTAMP_COL,
                                                                   self._SERIES_COL,
                                                                   self._ANOMALY_COL]),
                          inplace=True)
        
        if resample:
            if verbose:
                print_step("Resampling dataset to chosen granularity")
            
            self.dataset[self._TIMESTAMP_COL] = pd.to_datetime(self.dataset[self._TIMESTAMP_COL])
            self.dataset.index = pd.to_datetime(self.dataset[self._TIMESTAMP_COL])
            self.dataset = self.dataset.resample(resampling_granularity).agg({self._SERIES_COL: np.mean, self._ANOMALY_COL: np.max})
            self.dataset.reset_index(inplace=True)
            self.dataset[self._TIMESTAMP_COL] = self.dataset[self._TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if verbose:
            print_step("Dealing with missing values with specified strategy")
        
        if missing_strategy == MissingStrategy.DROP:
            self.dataset.dropna(inplace=True)
        elif missing_strategy == MissingStrategy.FIXED_VALUE:
            self.dataset.fillna(missing_fixed_value, inplace=True)
            
        if verbose:
            print_header("Ended dataset reading")
        
        return self
    
    def get_complete_dataframe(self) -> pd.DataFrame:
        """Same as reading, but all properties are returned (not only target).

        Returns
        -------
        pd.DataFrame
            Dataset with complete information.
        """
        if self._unmodified_dataset is None:
            raise RuntimeError("You must first read the dataset before being "
                               "able to get it.")
        
        enhanced_dataset = self._add_information(complete=True)
        
        new_dataset = self._unmodified_dataset.copy()
        new_dataset.insert(len(new_dataset.columns),
                           self._ANOMALY_COL,
                           enhanced_dataset[self._ANOMALY_COL].values)
        new_dataset.insert(len(new_dataset.columns),
                           self._ANOMALY_TYPE,
                           enhanced_dataset[self._ANOMALY_TYPE].values)
        new_dataset.insert(len(new_dataset.columns),
                           self._DAY_COL,
                           enhanced_dataset[self._DAY_COL].values)
        new_dataset = new_dataset.rename(columns={
            self.timestamp_col: self._TIMESTAMP_COL,
            self.univariate_col: self._SERIES_COL
        })
        new_dataset = new_dataset.drop(columns=new_dataset.columns.difference([self._TIMESTAMP_COL,
                                                                               self._SERIES_COL,
                                                                               self._ANOMALY_COL,
                                                                               self._ANOMALY_TYPE,
                                                                               self._DAY_COL]))
        
        return new_dataset
    
    def _add_information(self, complete: bool = False,
                         verbose: bool = False) -> pd.DataFrame:
        """Add information contained in the anomalies' path.
        
        Parameters
        ----------
        complete : bool, default=False
            States if anomaly columns must be added beside the labels.
        
        verbose: bool, default=False
            States if detailed printing must be done.
        
        Returns
        -------
        enhanced_dataset : pd.DataFrame
            The dataset enhanced with the information contained in the json file
            of the anomalies.
        """
        if verbose:
            print_step("Converting ODIN TS format to classical GT format")
            print_step("Reading ODIN TS anomalies json")
        
        # read the file with anomalies annotations
        anomalies_df = pd.read_csv(self.anomalies_path)

        # translate the dataset in a more accessible format for modification
        dataset_cp: pd.DataFrame = self._unmodified_dataset.copy()
        dataset_cp[self.timestamp_col] = pd.to_datetime(dataset_cp[self.timestamp_col],
                                                        format="%Y-%m-%d %H:%M:%S")
        dataset_cp = dataset_cp.set_index(self.timestamp_col)

        # add the columns with anomaly labels
        anomalies = np.zeros(dataset_cp.shape[0])
        day = np.ones(dataset_cp.shape[0]) * -1
        anomaly_type = ["No"] * dataset_cp.shape[0]
        dataset_cp.insert(len(dataset_cp.columns), self._ANOMALY_COL, anomalies)
        
        if complete:
            dataset_cp.insert(len(dataset_cp.columns), self._DAY_COL, day)
            dataset_cp.insert(len(dataset_cp.columns), self._ANOMALY_TYPE, anomaly_type)

        if verbose:
            print_step("Retrieving anomaly intervals")
            
        # get the anomaly intervals
        anomaly_intervals = [(datetime.strptime(el[0], "%Y-%m-%d %H:%M:%S"),
                              datetime.strptime(el[1], "%Y-%m-%d %H:%M:%S"))
                             for el in zip(anomalies_df[self.start_col].tolist(),
                                           anomalies_df[self.end_col].tolist())]
        if complete:
            anomaly_type_dict = {datetime.strptime(el[0], "%Y-%m-%d %H:%M:%S"): el[1]
                                 for el in zip(anomalies_df[self.start_col].tolist(),
                                               anomalies_df[self.anomaly_type_col].tolist())}

        if verbose:
            print_step("Converting intervals to labels for each point")

        # build the anomaly labels on original dataset
        for start, end in anomaly_intervals:
            dataset_cp.loc[start:end, self._ANOMALY_COL] = 1
            if complete:
                dataset_cp.loc[start:end, self._ANOMALY_TYPE] = anomaly_type_dict[start]
                for idx, row in dataset_cp.loc[start:end].iterrows():
                    dataset_cp.loc[idx, self._DAY_COL] = idx.to_pydatetime().weekday()
        
        if verbose:
            print_step("All intervals have been converted to point labels")
            print_step("Converted ODIN TS format")
        
        return dataset_cp
    