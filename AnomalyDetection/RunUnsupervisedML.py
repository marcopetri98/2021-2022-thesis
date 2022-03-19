import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN

DUMMIES = ["all_1", "all_0", "random"]
ALGORITHM = "dbscan"

DATASET = "nyc_taxi.csv"
DATASET_FOLDER = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
TRUTH_PREFIX = "truth"
ALL_METRICS = True

WINDOW = 30
STRIDE = 1

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

data = preprocess(np.array(test_data).reshape(test_data.shape[0], 1))
data_labels = test_labels
dataframe = test.copy()
dataframe["value"] = data

match ALGORITHM:
	case "dbscan":
		model = TimeSeriesAnomalyDBSCAN(window=30,
										eps=0.5,
										min_samples=15,
										metric="euclidean")
		model.fit(data)
	
	case "lof":
		pass

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
		labels = np.random.randint(0,2,num_pts)
		scores = labels == 1

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, False)
	make_metric_plots(dataframe, true_labels, scores, labels)
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
