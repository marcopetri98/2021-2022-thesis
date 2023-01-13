from __future__ import annotations
import abc
from abc import ABC

import numpy as np
import pandas as pd

from generator.exceptions.FlowError import FlowError


class SyntheticGenerator(ABC):
    """Interface for a dataset generator object.

    The SyntheticDatasetGenerator is a python abstract class (i.e. an interface)
    for the generation of a dataset as a pandas DataFrame. Each dataset
    generator implemented in this project is a SyntheticDatasetGenerator. The
    dataset generation comprehend the dimensions of the data points, the number
    of data points, the eventual target annotation in case of a supervised
    dataset generation (ground truth).

    Attributes
    ----------
    dataset : ndarray
        The numpy ndarray representing the dataset.

    dataset_frame : DataFrame
        The pandas dataframe representing the dataset.

    supervised : bool
        A boolean value representing if the dataset is supervised or not.

    ground_truth : ndarray
        It is the ground truth of the dataset in case the task is supervised.

    labels : list[str]
        Labels to be used for the dataset.
    """
    def __init__(self, supervised: bool,
                 labels: list[str]):
        super().__init__()
        self.dataset = None
        self.dataset_frame = None
        self.supervised = supervised
        self.ground_truth = None
        self.labels = labels.copy()

    @abc.abstractmethod
    def generate(self, *args,
                 **kwargs) -> SyntheticGenerator:
        """Generates the dataset.

        This method generates a synthetic dataset depending on the attributes
        and attributes of the class and of the parameters that have been passed
        to the function. The parameters passed to this function are specific for
        the dataset to be generated, whereas the attributes of the class are
        specific of the class of datasets that the object is capable of
        generating.

        Parameters
        ----------
        args
            Not used, present to allow signature change in subclasses.

        kwargs
            Not used, present to allow signature change in subclasses.

        Returns
        -------
        generator : SyntheticGenerator
            A reference to itself to allow call concatenation.
        """
        pass

    def get_dataset(self, *args,
                    **kwargs) -> np.ndarray:
        """Gets a copy of the numpy array of the dataset.

        Parameters
        ----------
        args
            Not used, present to allow signature change in subclasses.

        kwargs
            Not used, present to allow signature change in subclasses.

        Returns
        -------
        numpy_dataset : ndarray
            The numpy array of the dataset without the index.
        """
        if self.dataset is not None:
            return self.dataset.copy()
        else:
            raise FlowError("You must first generate the dataset before being "
                            "able to get it.")

    def get_dataframe(self, *args,
                      **kwargs) -> pd.DataFrame:
        """Gets a copy of the dataframe of the dataset.

        Parameters
        ----------
        args
            Not used, present to allow signature change in subclasses.

        kwargs
            Not used, present to allow signature change in subclasses.

        Returns
        -------
        dataset_dataframe : DataFrame
            The pandas dataframe of the dataset.
        """
        if self.dataset_frame is not None:
            return self.dataset_frame.copy()
        else:
            raise FlowError("You must first generate the dataframe before being "
                            "able to get it.")

    def get_ground_truth(self, *args,
                         **kwargs) -> np.ndarray:
        """Gets the ground truth for the dataset.

        Parameters
        ----------
        args
            Not used, present to allow signature change in subclasses.

        kwargs
            Not used, present to allow signature change in subclasses.

        Returns
        -------
        ground_truth : ndarray
            The ground truth for the dataset
        """
        if self.ground_truth is not None:
            return self.ground_truth.copy()
        else:
            raise FlowError("You must first generate the ground truth before "
                            "being able to get it.")
