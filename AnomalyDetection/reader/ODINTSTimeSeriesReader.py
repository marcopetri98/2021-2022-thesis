from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from reader.TimeSeriesReader import TimeSeriesReader


class ODINTSTimeSeriesReader(TimeSeriesReader):
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
    
    def __init__(self, anomalies_path: str,
                 timestamp_col: str,
                 univariate_col: str,
                 start_col: str = "start_date",
                 end_col: str = "end_date"):
        super().__init__()
        
        self.anomalies_path = anomalies_path
        self.timestamp_col = timestamp_col
        self.univariate_col = univariate_col
        self.start_col = start_col
        self.end_col = end_col
        
    def read(self, path: str,
			 file_format: str = "csv") -> ODINTSTimeSeriesReader:
        super().read(path, file_format)
        
        # read the file with anomalies annotations
        anomalies_df = pd.read_csv(self.anomalies_path)
        
        # translate the dataset in a more accessible format for modification
        dataset_cp: pd.DataFrame = self.dataset.copy()
        dataset_cp[self.timestamp_col] = pd.to_datetime(dataset_cp[self.timestamp_col],
                                                        format="%Y-%m-%d %H:%M:%S")
        dataset_cp = dataset_cp.set_index(self.timestamp_col)
        dataset_cp = dataset_cp.drop(columns=dataset_cp.columns.difference([self.univariate_col]))
        
        # add the columns with anomaly labels
        anomalies = np.zeros(dataset_cp.shape[0])
        dataset_cp.insert(len(dataset_cp.columns), self._ANOMALY_COL, anomalies)
        
        # get the anomaly intervals
        anomaly_intervals = [(datetime.strptime(el[0], "%Y-%m-%d %H:%M:%S"),
                              datetime.strptime(el[1], "%Y-%m-%d %H:%M:%S"))
                             for el in zip(anomalies_df[self.start_col].tolist(),
                                           anomalies_df[self.end_col].tolist())]
        
        # build the anomaly labels on original dataset
        for start, end in anomaly_intervals:
            dataset_cp.loc[start:end, self._ANOMALY_COL] = 1
            
        # add anomaly labels to original dataset and drop useless columns
        self.dataset.insert(len(self.dataset.columns),
                            self._ANOMALY_COL,
                            dataset_cp[self._ANOMALY_COL].values)
        self.dataset = self.dataset.rename(columns={
            self.timestamp_col : self._TIMESTAMP_COL,
            self.univariate_col : self._SERIES_COL
        })
        self.dataset = self.dataset.drop(columns=self.dataset.columns.difference([self._TIMESTAMP_COL,
                                                                                  self._SERIES_COL,
                                                                                  self._ANOMALY_COL]))
        
        return self
    