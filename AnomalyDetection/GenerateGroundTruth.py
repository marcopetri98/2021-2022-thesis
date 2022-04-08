import datetime
import json

import pandas as pd
import numpy as np

DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training_dl/"
TESTING_PATH = DATASET_PATH + "testing_dl/"
ANNOTATED_PATH = DATASET_PATH + "annotated_dl/"
PURE_DATA = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_TRUTHS_PATH = "dataset/combined_labels.json"
GROUND_WINDOWS_PATH = "dataset/combined_windows.json"
NO_WINDOW = True

# Determine path of the dataset
pure_data_path = DATASET_PATH + PURE_DATA
if NO_WINDOW:
	supervised_data_path = ANNOTATED_PATH + PURE_DATA
else:
	supervised_data_path = ANNOTATED_PATH + "detailed_" + PURE_DATA

# Get the ground truth definition by windows
file = open(GROUND_WINDOWS_PATH)
combined_windows = json.load(file)
file.close()

file = open(GROUND_TRUTHS_PATH)
combined_labels = json.load(file)
file.close()

# Get the ground truth of the desired dataset
desired_windows = combined_windows[PURE_DATA_KEY]
anomalies = combined_labels[PURE_DATA_KEY]

# Generate the dataset with ground truth
original_dataset = pd.read_csv(pure_data_path)
timestamps = original_dataset["timestamp"]
ground_truth = [0] * original_dataset.shape[0]
for i in range(original_dataset.shape[0]):
	timestamp = datetime.datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
	
	if not NO_WINDOW:
		for window in desired_windows:
			first = datetime.datetime.strptime(window[0], "%Y-%m-%d %H:%M:%S.%f")
			last = datetime.datetime.strptime(window[1], "%Y-%m-%d %H:%M:%S.%f")
			if first <= timestamp <= last:
				ground_truth[i] = 2
				break
	
	if timestamps[i] in anomalies:
		ground_truth[i] = 1

# Convert into numpy array the ground truth and add it to the dataframe, then,
# save it to csv
truth = np.array(ground_truth)
original_dataset.insert(len(original_dataset.columns),
						"target",
						truth)
original_dataset.to_csv(supervised_data_path)
