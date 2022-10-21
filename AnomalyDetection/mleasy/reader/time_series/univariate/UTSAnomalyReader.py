from typing import Tuple

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

    def get_train_valid_test_ground_truth(self, col_name: str,
                                          *args,
                                          **kwargs) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray]:
        """Get the ground truth for training, validation and testing.

        Parameters
        ----------
        col_name : str
            The column name of the ground truth.

        args
            Not used, present for inheritance change of signature.

        kwargs
            Not used, present for inheritance change of signature.

        Returns
        -------
        train_gt, valid_gt, test_gt : ndarray, ndarray, ndarray
            The ground truths relative to training, validation and testing
            frames.
        """
        check_not_default_attributes(self, {"dataset": None})

        if col_name not in self.dataset.columns:
            raise ValueError("The column specified does not exist")

        train_gt = self.train_frame[col_name].values
        valid_gt = self.valid_frame[col_name].values
        test_gt = self.test_frame[col_name].values

        return train_gt, valid_gt, test_gt
