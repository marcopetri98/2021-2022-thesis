import json
import datetime

import numpy as np
import pandas as pd


def get_windows_indices(dataframe: pd.DataFrame,
						data_key: str,
						combined_windows_path: str,
						index_column: str = "timestamp") -> np.ndarray:
	"""Gets an array with the indices of the windows borders.

	Parameters
	----------
	dataframe : DataFrame
		The dataframe where to search the indices of the windows.
	data_key : str
		The key on the combined windows json file.
	combined_windows_path : str
		The path where it is possible to find the combined windows.
	index_column : str
		The column where the index is stored.

	Returns
	-------
	window_borders: ndarray
		The array with the indices of the borders of the windows.
	"""
	file = open(combined_windows_path)
	combined_windows = json.load(file)
	file.close()

	desired_windows = combined_windows[data_key]
	desired_windows = [elem for window in desired_windows for elem in window]
	desired_windows = [elem[:-7] for elem in desired_windows]
	timestamps = dataframe[index_column]
	mask = np.zeros(np.array(dataframe[index_column]).size)
	for i in range(np.array(timestamps).size):
		if timestamps[i] in desired_windows:
			mask[i] = 1
	bool_mask = [True if val == 1 else False for val in mask]

	return np.argwhere(bool_mask)
