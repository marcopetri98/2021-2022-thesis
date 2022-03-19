# Python imports
import datetime

# External imports
import json
import pandas as pd
import numpy as np

# Project imports

DATASET_PATH = "dataset/"
PURE_DATA = "nyc_taxi.csv"
PURE_DATA_KEY = "realKnownCause/nyc_taxi.csv"
GROUND_TRUTHS_PATH = "dataset/combined_windows.json"

# Determine path of the dataset
pure_data_path = DATASET_PATH + PURE_DATA
supervised_data_path = DATASET_PATH + "truth_" + PURE_DATA

# Get the ground truth definition by windows
file = open(GROUND_TRUTHS_PATH)
combined_windows = json.load(file)
file.close()

# Get the ground truth of the desired dataset
desired_windows = combined_windows[PURE_DATA_KEY]

# Generate the dataset with ground truth
original_dataset = pd.read_csv(pure_data_path)
timestamps = original_dataset["timestamp"]
ground_truth = [0] * original_dataset.shape[0]
for i in range(original_dataset.shape[0]):
	is_anomaly = False
	timestamp = datetime.datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
	
	for window in desired_windows:
		first = datetime.datetime.strptime(window[0], "%Y-%m-%d %H:%M:%S.%f")
		last = datetime.datetime.strptime(window[1], "%Y-%m-%d %H:%M:%S.%f")
		if first <= timestamp <= last:
			is_anomaly = True
			break
			
	if is_anomaly:
		ground_truth[i] = 1

# Convert into numpy array the ground truth and add it to the dataframe, then,
# save it to csv
truth = np.array(ground_truth)
original_dataset.insert(len(original_dataset.columns),
						"target",
						truth)
original_dataset.to_csv(supervised_data_path)
