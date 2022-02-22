import pandas as pd

from typing import Callable
from abc import ABC
from generator.FlowError import FlowError


class SyntheticDatasetGenerator(ABC):
	"""Interface for a dataset generator object.
	
	The SyntheticDatasetGenerator is a python abstract class (i.e. an interface)
	for the generation of a dataset as a pandas DataFrame. Each dataset
	generator implemented in this project is a SyntheticDatasetGenerator. The
	dataset generation comprehend the dimensions of the data points, the number
	of data points, the eventual target annotation in case of a supervised
	dataset generation (ground truth).
	"""
	def __init__(self, supervised : bool,
				 labels : list[str]):
		super().__init__()
		self.dataset = None
		self.supervised = supervised
		self.labels = labels.copy()
	
	def generate(self, num_points : int,
				 dimensions : int,
				 noise : list[str],
				 custom_noise : list[Callable[[], float]] = None,
				 verbose : bool = True) -> pd.DataFrame:
		pass
	
	def get_dataset(self):
		if self.dataset is not None:
			return self.dataset
		else:
			raise FlowError("You must first generate the dataset before being "
							"able to get it.")