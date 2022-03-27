import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from models.anomaly.deep_learning.TimeSeriesAnomalyGRU import \
	TimeSeriesAnomalyGRU
from models.anomaly.deep_learning.TimeSeriesAnomalyLSTM import TimeSeriesAnomalyLSTM
from models.anomaly.deep_learning.TimeSeriesAnomalyLSTMAutoencoder import \
	TimeSeriesAnomalyLSTMAutoencoder

DUMMIES = ["all_1", "all_0", "random"]
ALGORITHM = "lstm autoencoder"

VALIDATION_DIM = 0.2
DATASET = "ambient_temperature_system_failure.csv"
DATASET_FOLDER = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
TRUTH_PREFIX = "truth"
ALL_METRICS = True
LOAD_MODEL = False
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

# Create the slices of the trainig data
validation_elems = data.shape[0] * VALIDATION_DIM

change_idx = np.where(np.array(data_labels[:-1]) != np.array(data_labels[1:]))
change_idx = np.array(change_idx) + 1
change_idx = change_idx.reshape(change_idx.shape[1])
change_idx = np.concatenate((np.array([0]), change_idx))
normal_slices = []
anomaly_slices = []
for i in range(len(change_idx)):
	start = change_idx[i]
	if i < len(change_idx) -1:
		stop = change_idx[i+1]
	else:
		stop = data.shape[0]
	
	if data_labels[change_idx[i]] == 1:
		anomaly_slices.append(slice(start, stop))
	else:
		normal_slices.append(slice(start, stop))

training_slices = [slice(0, 3134), slice(4312, 5814)]
validation_slices = [slice(3140, 4302)]

match ALGORITHM:
	case "lstm":
		model = TimeSeriesAnomalyLSTM(window=30,
									  max_epochs=50,
									  batch_size=32,
									  batch_divide_training=True,
									  filename="lstm_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru":
		model = TimeSeriesAnomalyGRU(window=30,
									 max_epochs=50,
									 batch_size=32,
									 batch_divide_training=True,
									 filename="gru_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/gru_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn":
		model = TimeSeriesAnomalyGRU(window=30,
									 max_epochs=50,
									 batch_size=32,
									 batch_divide_training=True,
									 filename="cnn_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "lstm autoencoder":
		model = TimeSeriesAnomalyLSTMAutoencoder(window=32,
												 max_epochs=50,
												 batch_size=32,
												 filename="lstm_ae_paper",
												 extend_not_multiple=True)
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_ae_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru autoencoder":
		model = TimeSeriesAnomalyGRU(window=30,
									 max_epochs=50,
									 batch_size=32,
									 filename="gru_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/gru_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn autoencoder":
		model = TimeSeriesAnomalyGRU(window=30,
									 max_epochs=50,
									 batch_size=32,
									 filename="cnn_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)

true_labels = data_test_labels
if ALGORITHM not in DUMMIES:
	labels = model.predict(data, data_test)
	scores = model.anomaly_score(data, data_test)
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
