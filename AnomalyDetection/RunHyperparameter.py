import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer

from models.time_series.anomaly import TimeSeriesAnomalyLOF
from tuning.hyperparameter.GaussianProcessesSearch import GaussianProcessesSearch

DATASET = "ambient_temperature_system_failure.csv"
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training/"
UNSUPERVISED = False
SELF_SUPERVISED = True
TRAIN = False

training = pd.read_csv(TRAINING_PATH + DATASET)
training_timestamps = training["timestamp"]
training_data = training["value"]
training_labels = training["target"]

data = StandardScaler().fit_transform(np.array(training_data).reshape(training_data.shape[0], 1))
data_labels = training_labels
dataframe = training.copy()
dataframe["value"] = data

training_slices = [slice(0, 3600), slice(3900, 5814)]
validation_slices = [slice(3600, 3900)]

train = None
train_labels = None
for slice_ in training_slices:
	if train is None:
		train = data[slice_]
		train_labels = data_labels[slice_]
	else:
		train = np.concatenate([train, data[slice_]], axis=0)
		train_labels = np.concatenate([train_labels, data_labels[slice_]], axis=0)

hyper_searcher = GaussianProcessesSearch(TimeSeriesAnomalyLOF(classification="points_score",
															  anomaly_contamination=0.01),
										 [
										  Integer(2, 100, name="window"),
										  # Integer(1, 20, name="stride"),
										  # Real(0.0, 1.0, name="anomaly_threshold"),
										  # Real(0.00001, 0.5, prior="log-uniform", name="contamination"),
										  Integer(2, 100, name="n_estimators"),
										  Integer(2, 50, name="max_samples")
									  ],
									  "searches/lof/",
										 "temperature_window_neighbors_gaussian_2",
										 80,
										 10,
										 train_and_test=True,
										 load_checkpoint=False)
if TRAIN:
	if UNSUPERVISED:
		hyper_searcher.fit(data, data_labels)
	elif SELF_SUPERVISED:
		hyper_searcher.fit(train, train_labels)

hyper_searcher.print_search()
