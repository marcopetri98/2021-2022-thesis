import numpy as np
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from get_windows_indices import get_windows_indices
import visualizer.Viewer as vw
from models.time_series.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.time_series.anomaly.TimeSeriesAnomalyIForest import TimeSeriesAnomalyIForest
from models.time_series.anomaly.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF
from models.time_series.anomaly.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM
from models.time_series.anomaly.TimeSeriesAnomalyOSVMPhase import TimeSeriesAnomalyOSVMPhase
from reader.NABTimeSeriesReader import NABTimeSeriesReader

#################################
#								#
#								#
#		RUN CONSTANTS			#
#								#
#								#
#################################

ALGORITHM = "lof"

DATASET_PATH = "dataset/"
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "dataset/combined_windows.json"
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

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train_perc=0.8).get_train_test_dataframes()

#################################
#								#
#								#
#			DEFINE DATA			#
#								#
#								#
#################################
# Data used to train
data = preprocess(np.array(training["value"]).reshape(training["value"].shape[0], 1))
data_labels = training["target"]

# Data used to test
data_test = preprocess(np.array(test["value"]).reshape(test["value"].shape[0], 1))
data_test_labels = test["target"]

# Data used to evaluate
dataframe = test.copy()
dataframe["value"] = test["value"]

training_slices = [slice(0, 3600), slice(3900, 5814)]
validation_slices = [slice(3600, 3900)]

train = None
for slice_ in training_slices:
	if train is None:
		train = data[slice_]
	else:
		train = np.concatenate([train, data[slice_]], axis=0)

if CHECK_OVERFITTING:
	data_test = preprocess(np.array(training["value"]).reshape(training["value"].shape[0], 1))
	data_test_labels = training["target"]
	dataframe = training.copy()
	dataframe["value"] = data_test
elif ALL_DATA:
	data_test = preprocess(np.array(all_df["value"]).reshape(all_df["value"].shape[0], 1))
	data_test_labels = all_df["target"]
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
								  anomaly_portion=0.0015)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "phase osvm":
	model = TimeSeriesAnomalyOSVMPhase(windows=[5, 10, 20],
									   nu=0.4,
									   classification="points_score",
									   anomaly_portion=0.0015)
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
	compute_metrics(true_labels, scores, labels, only_roc_auc=False)
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
