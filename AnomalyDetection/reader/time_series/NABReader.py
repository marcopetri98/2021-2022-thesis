from __future__ import annotations

import datetime
import json

import numpy as np

from reader.time_series.UTSReader import UTSReader
from utils.printing import print_header, print_step


class NABReader(UTSReader):
	"""A reader of NAB time series datasets.
	
	Parameters
	----------
	label_path : str
		The path to the folder in which the combined labels and the combined
		windows of the NAB dataset are stored.
		
	labels_name : str, default="combined_labels.json"
		The filename of the combined labels of NAB dataset.
		
	window_name : str, default="combined_windows.json"
		The filename of the combined windows of NAB dataset.
		
	save_windows : bool, default=False
		Whether we should save windows and not only the anomalous point in the
		extracted dataset.
	"""
	
	def __init__(self, label_path: str,
				 labels_name: str = "combined_labels.json",
				 window_name: str = "combined_windows.json",
				 save_windows: bool = False):
		super().__init__()
		
		self.label_path = label_path
		self.labels_name = labels_name
		self.window_name = window_name
		self.save_windows = save_windows
		self.combined_labels = {}
		self.combined_windows = {}

	def read(self, path: str,
			 file_format: str = "csv",
			 verbose: bool = True,
			 *args,
			 **kwargs) -> NABReader:
		if "/" in path:
			sep = "/"
		else:
			sep = "\\"

		if verbose:
			print_header("Start reading dataset")

		# Get the dataset filename
		dataset_file = path.split(sep)[-1]
		
		# Gets the dictionaries of combined labels and windows
		if verbose:
			print_step("Start to read combined labels")
			
		file = open(self.label_path + self.labels_name)
		self.combined_labels : dict = json.load(file)
		file.close()
		
		if verbose:
			print_step("Ended combined labels reading")
			print_step("Start to read combined windows")
		
		file = open(self.label_path + self.window_name)
		self.combined_windows : dict = json.load(file)
		file.close()
		
		if verbose:
			print_step("Ended combined windows reading")
		
		# Gets the key of the dataset
		dataset_key = list(filter(lambda x: dataset_file in x, self.combined_labels.keys()))
		dataset_key = dataset_key[0]
		
		# Get the ground truth of the desired dataset
		windows = self.combined_windows[dataset_key]
		labels = self.combined_labels[dataset_key]
		
		if verbose:
			print_step("Start reading the dataset values")
		
		# Generate the dataset with ground truth
		super().read(path=path, file_format=file_format, verbose=False)
		
		if verbose:
			print_step("Ended dataset reading")
			print_step("Converting NAB format to classical GT format")
		
		timestamps = self.dataset[self._TIMESTAMP_COL]
		# TODO: directly create an ndarray
		ground_truth = [0] * self.dataset.shape[0]
		for i in range(self.dataset.shape[0]):
			timestamp = datetime.datetime.strptime(timestamps[i],
												   "%Y-%m-%d %H:%M:%S")
			
			if self.save_windows:
				for window in windows:
					first = datetime.datetime.strptime(window[0],
													   "%Y-%m-%d %H:%M:%S.%f")
					last = datetime.datetime.strptime(window[1],
													  "%Y-%m-%d %H:%M:%S.%f")
					if first <= timestamp <= last:
						ground_truth[i] = 2
						break
			
			if timestamps[i] in labels:
				ground_truth[i] = 1
		
		# Convert into numpy array the ground truth and add it to the dataframe
		truth = np.array(ground_truth)
		self.dataset.insert(len(self.dataset.columns),
							self._ANOMALY_COL,
							truth)
		
		if verbose:
			print_step("Ended converting NAB format to classical one")
		
		return self
