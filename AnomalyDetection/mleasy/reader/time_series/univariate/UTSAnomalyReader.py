import numpy as np

from mleasy.input_validation import check_not_default_attributes
from mleasy.reader.time_series.univariate import UTSReader
from mleasy.reader import IDataSupervised


class UTSAnomalyReader(UTSReader, IDataSupervised):
    """A reader for univariate time series anomaly dataset.
    """
    def get_ground_truth(self, col_name: str, *args, **kwargs) -> np.ndarray:
        check_not_default_attributes(self, {"dataset": None})
    
        if col_name not in self.dataset.columns:
            raise ValueError("The column specified does not exist")
    
        targets = self.dataset[col_name]
        return np.array(targets)
