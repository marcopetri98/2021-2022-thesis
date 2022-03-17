# Python imports

# External imports

# Project imports
import pandas as pd

PERC_TRAIN = 0.8
DATASET_PATH = "dataset/"
PURE_DATA = "ambient_temperature_system_failure.csv"
LABELLED_DATA = "truth_ambient_temperature_system_failure.csv"

# Determine path of the dataset
labelled_data_path = DATASET_PATH + LABELLED_DATA
training_data_path = DATASET_PATH + "training_" + PURE_DATA
testing_data_path = DATASET_PATH + "test_" + PURE_DATA

# Split in training and test
labelled_dataset = pd.read_csv(labelled_data_path)
num_of_test = int((1 - PERC_TRAIN) * labelled_dataset.shape[0])
training = labelled_dataset[0:-num_of_test]
testing = labelled_dataset[-num_of_test:]

training.to_csv(training_data_path)
testing.to_csv(testing_data_path)
