import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.anomaly.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF

DUMMIES = ["all_1", "all_0", "random"]
ALGORITHM = "dbscan"

DATASET = "ambient_temperature_system_failure.csv"
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training/"
TESTING_PATH = DATASET_PATH + "testing/"
ANNOTATED_PATH = DATASET_PATH + "annotated/"
ALL_METRICS = True
CHECK_OVERFITTING = False


def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)


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

data = preprocess(np.array(test_data).reshape(test_data.shape[0], 1))
data_labels = test_labels
dataframe = test.copy()
dataframe["value"] = data

if CHECK_OVERFITTING:
	data = preprocess(
		np.array(training_data).reshape(training_data.shape[0], 1))
	data_labels = training_labels
	dataframe = training.copy()
	dataframe["value"] = data

match ALGORITHM:
	case "dbscan":
		model = TimeSeriesAnomalyDBSCAN(window=3,
										eps=0.1,
										min_samples=5)
		model.fit(data)

	case "lof":
		model = TimeSeriesAnomalyLOF(window=170,
									 n_neighbors=199)
		model.fit(data)

true_labels = data_labels
if ALGORITHM not in DUMMIES:
	labels = model.labels_
	scores = model.scores_
else:
	# With all_1 all are categorized as anomalies, with all_0 all the
	# points are categorized as being normal. With random all the points are
	# randomly drawn
	num_pts = data.shape[0]
	if ALGORITHM == "all_1":
		labels = np.ones(num_pts)
		scores = np.ones(num_pts)
	elif ALGORITHM == "all_0":
		labels = np.zeros(num_pts)
		scores = np.zeros(num_pts)
	else:
		labels = np.random.randint(0, 2, num_pts)
		scores = labels == 1

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, False)
	make_metric_plots(dataframe, true_labels, scores, labels)
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
