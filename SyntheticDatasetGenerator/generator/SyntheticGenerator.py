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

	* dataset: the numpy ndarray representing the dataset.
	* dataset_frame: the pandas dataframe representing the dataset.
	* supervised: a boolean value representing if the dataset is supervised or
	not.
	* labels: labels to be used for the dataset.
	"""
	def __init__(self, supervised : bool,
				 labels : list[str]):
		super().__init__()
		self.dataset = None
		self.dataset_frame = None
		self.supervised = supervised
		self.labels = labels.copy()
	
	def get_dataset(self) -> np.ndarray:
		if self.dataset is not None:
			return self.dataset.copy()
		else:
			raise FlowError("You must first generate the dataset before being "
							"able to get it.")
	
	def get_dataframe(self) -> pd.DataFrame:
		if self.dataset_frame is not None:
			return  self.dataset_frame.copy()
		else:
			raise FlowError("You must first generate the dataframe before being "
							"able to get it.")