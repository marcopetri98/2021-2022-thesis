import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from get_windows_indices import get_windows_indices
import visualizer.Viewer as vw
from models.time_series.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.time_series.anomaly.TimeSeriesAnomalyIForest import TimeSeriesAnomalyIForest
from models.time_series.anomaly.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF
from models.time_series.anomaly.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM

#################################
#								#
#								#
#		RUN CONSTANTS			#
#								#
#								#
#################################
ALGORITHM = "lof"

DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "dataset/combined_windows.json"
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training/"
TESTING_PATH = DATASET_PATH + "testing/"
ANNOTATED_PATH = DATASET_PATH + "annotated/"
ALL_METRICS = True
CHECK_OVERFITTING = False
ALL_DATA = True
UNSUPERVISED = False
SELF_SUPERVISED = True


def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

#################################
#								#
#								#
#			LOAD DATA			#
#								#
#								#
#################################
model = None

all_df = pd.read_csv(ANNOTATED_PATH + DATASET)
all_timestamps = all_df["timestamp"]
all_data = all_df["value"]
all_labels = all_df["target"]

training = pd.read_csv(TRAINING_PATH + DATASET)
training_timestamps = training["timestamp"]
training_data = training["value"]
training_labels = training["target"]

test = pd.read_csv(TESTING_PATH + DATASET)
test_timestamps = test["timestamp"]
test_data = test["value"]
test_labels = test["target"]

#################################
#								#
#								#
#			DEFINE DATA			#
#								#
#								#
#################################
# Data used to train
data = preprocess(np.array(training_data).reshape(training_data.shape[0], 1))
data_labels = training_labels

# Data used to test
data_test = preprocess(np.array(test_data).reshape(test_data.shape[0], 1))
data_test_labels = test_labels

# Data used to evaluate
dataframe = test.copy()
dataframe["value"] = test_data

training_slices = [slice(0, 3600), slice(3900, 5814)]
validation_slices = [slice(3600, 3900)]

train = None
for slice_ in training_slices:
	if train is None:
		train = data[slice_]
	else:
		train = np.concatenate([train, data[slice_]], axis=0)

if CHECK_OVERFITTING:
	data_test = preprocess(np.array(training_data).reshape(training_data.shape[0], 1))
	data_test_labels = training_labels
	dataframe = training.copy()
	dataframe["value"] = data_test
elif ALL_DATA:
	data_test = preprocess(np.array(all_data).reshape(all_data.shape[0], 1))
	data_test_labels = all_labels
	dataframe = all_df.copy()
	dataframe["value"] = data_test

#################################
#								#
#								#
#			FIT MODEL			#
#								#
#								#
#################################
if UNSUPERVISED and ALGORITHM == "dbscan":
	model = TimeSeriesAnomalyDBSCAN(window=3,
									eps=2.75,
									min_samples=31,
									#anomaly_threshold=0.9888,
									anomaly_portion=0.0003,
									classification="points_score")
elif UNSUPERVISED and ALGORITHM == "lof":
	# Better window=3, neighbors=40
	model = TimeSeriesAnomalyLOF(window=3,
								 classification="points_score",
								 anomaly_portion=0.01,
								 n_neighbors=100)
elif SELF_SUPERVISED and ALGORITHM == "lof":
	# Better window=3, neighbors=40
	model = TimeSeriesAnomalyLOF(window=3,
								 classification="points_score",
								 n_neighbors=40,
								 novelty=True)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "osvm":
	model = TimeSeriesAnomalyOSVM(window=10,
								  nu=0.4,
								  classification="points_score",
								  anomaly_contamination=0.0015)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "iforest":
	model = TimeSeriesAnomalyIForest(window=5,
									 # n_estimators=100,
									 contamination=0.004,
									 # max_samples=30,
									 random_state=22)
	model.fit(train)

#################################
#								#
#								#
#			EVALUATE			#
#								#
#								#
#################################
true_labels = data_test_labels
labels = model.classify(data_test)
scores = model.anomaly_score(data_test)

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	bars = get_windows_indices(all_df,
							   PURE_DATA_KEY,
							   GROUND_WINDOWS_PATH)
	all_timestamps = all_df["timestamp"].tolist()
	bars = [dataframe["timestamp"].tolist().index(all_timestamps[int(bar)])
			for bar in bars
			if all_timestamps[int(bar)] in dataframe["timestamp"].tolist()]
	bars = np.array(bars)
	vw.plot_time_series_with_predicitons_bars(dataframe,
											  labels,
											  bars,
											  pred_color='r')
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
