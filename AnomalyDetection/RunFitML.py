import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from models.anomaly.TimeSeriesAnomalyIForest import TimeSeriesAnomalyIForest
from models.anomaly.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM

DUMMIES = ["all_1", "all_0", "random"]
ALGORITHM = "iforest"

DATASET = "ambient_temperature_system_failure.csv"
DATASET_FOLDER = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
TRUTH_PREFIX = "truth"
ALL_METRICS = True
CHECK_OVERFITTING = False

def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)


all = pd.read_csv(DATASET_FOLDER + "truth_" + DATASET)
all_timestamps = all["timestamp"]
all_data = all["value"]
all_labels = all["target"]

training = pd.read_csv(DATASET_FOLDER + TRAINING_PREFIX + DATASET)
training_timestamps = training["timestamp"]
training_data = training["value"]
training_labels = training["target"]

test = pd.read_csv(DATASET_FOLDER + TESTING_PREFIX + DATASET)
test_timestamps = test["timestamp"]
test_data = test["value"]
test_labels = test["target"]

# Data used to train
data = preprocess(np.array(training_data).reshape(training_data.shape[0], 1))
data_labels = training_labels

# Data used to test
data_test = preprocess(np.array(test_data).reshape(test_data.shape[0], 1))
data_test_labels = test_labels

# Data used to evaluate
dataframe = test.copy()
dataframe["value"] = data_test

if CHECK_OVERFITTING:
	data_test = preprocess(np.array(training_data).reshape(training_data.shape[0], 1))
	data_test_labels = training_labels
	dataframe = training.copy()
	dataframe["value"] = data_test

match ALGORITHM:
	case "osvm":
		model = TimeSeriesAnomalyOSVM(window=22,
									  nu=0.97)
		model.fit(data, data_labels)
		
	case "iforest":
		model = TimeSeriesAnomalyIForest(window=240,
										 n_estimators=100,
										 max_samples=30,
										 random_state=22)
		model.fit(data)

true_labels = data_test_labels
if ALGORITHM not in DUMMIES:
	labels = model.predict(data_test)
	scores = model.anomaly_score(data_test)
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
