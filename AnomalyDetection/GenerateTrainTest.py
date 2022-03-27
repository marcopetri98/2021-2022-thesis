# Python imports

# External imports

# Project imports
import pandas as pd

PERC_TRAIN = 0.8
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training/"
TESTING_PATH = DATASET_PATH + "testing/"
ANNOTATED_PATH = DATASET_PATH + "annotated/"
PURE_DATA = "ambient_temperature_system_failure.csv"
DETAILED_DATA = "detailed_" + PURE_DATA
NO_WINDOW = True

# Determine path of the dataset
if NO_WINDOW:
	labelled_data_path = ANNOTATED_PATH + PURE_DATA
	training_data_path = TRAINING_PATH + PURE_DATA
	testing_data_path = TESTING_PATH + PURE_DATA
else:
	labelled_data_path = ANNOTATED_PATH + DETAILED_DATA
	training_data_path = TRAINING_PATH + DETAILED_DATA
	testing_data_path = TESTING_PATH + DETAILED_DATA

# Split in training and test
labelled_dataset = pd.read_csv(labelled_data_path)
num_of_test = int((1 - PERC_TRAIN) * labelled_dataset.shape[0])
training = labelled_dataset[0:-num_of_test]
testing = labelled_dataset[-num_of_test:]

training.to_csv(training_data_path)
testing.to_csv(testing_data_path)
