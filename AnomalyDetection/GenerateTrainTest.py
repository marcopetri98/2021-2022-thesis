# Python imports

# External imports

# Project imports
import pandas as pd

PERC_TRAIN = 0.8
DATASET_PATH = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
PURE_DATA = "nyc_taxi.csv"
LABELLED_DATA = "truth_nyc_taxi.csv"

# Determine path of the dataset
labelled_data_path = DATASET_PATH + LABELLED_DATA
training_data_path = DATASET_PATH + TRAINING_PREFIX + PURE_DATA
testing_data_path = DATASET_PATH + TESTING_PREFIX + PURE_DATA

# Split in training and test
labelled_dataset = pd.read_csv(labelled_data_path)
num_of_test = int((1 - PERC_TRAIN) * labelled_dataset.shape[0])
training = labelled_dataset[0:-num_of_test]
testing = labelled_dataset[-num_of_test:]

training.to_csv(training_data_path)
testing.to_csv(testing_data_path)
