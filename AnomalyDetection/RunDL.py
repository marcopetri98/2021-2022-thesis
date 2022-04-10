import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from models.time_series.anomaly.deep_learning.BraeiDenseAutoencoder import \
	BraeiDenseAutoencoder
from models.time_series.anomaly.deep_learning.BraeiGRU import BraeiGRU
from models.time_series.anomaly.deep_learning.BraeiLSTM import BraeiLSTM
from models.time_series.anomaly.deep_learning.TimeSeriesAnomalyLSTMAutoencoder import TimeSeriesAnomalyLSTMAutoencoder
from visualizer.Viewer import plot_time_series_forecast

ALGORITHM = "dense autoencoder"

VALIDATION_DIM = 0.2
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "dataset/combined_windows.json"
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training_dl/"
TESTING_PATH = DATASET_PATH + "testing_dl/"
ANNOTATED_PATH = DATASET_PATH + "annotated_dl/"
ALL_METRICS = True
LOAD_MODEL = False
CHECK_OVERFITTING = False
AUTOENCODER = True
AUTOENCODER_WINDOW = 30


def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)


#################################
#								#
#								#
#			LOAD DATA			#
#								#
#								#
#################################
np.random.seed(57)
tf.random.set_seed(57)
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

if AUTOENCODER:
	if data_test.shape[0] % AUTOENCODER_WINDOW != 0:
		remainder_points = data_test.shape[0] % AUTOENCODER_WINDOW
		
		np_data = np.array(data)
		np_data_test = np.array(data_test)
		np_data_labels = np.array(data_labels)
		np_data_test_labels = np.array(data_test_labels)
		data = np.concatenate((np_data, np_data_test[0:remainder_points]))
		data_labels = np.concatenate((np_data_labels, np_data_test_labels[0:remainder_points]))
		data_test = data_test[remainder_points:]
		data_test_labels = data_test_labels[remainder_points:]
		
		np_train = np.concatenate((np.array(training), np.array(test[:remainder_points])))
		training = pd.DataFrame(np_train, columns=training.columns)
		training_timestamps = training["timestamp"]
		training_data = training["value"]
		training_labels = training["target"]
		
		np_test = np.array(test[remainder_points:])
		test = pd.DataFrame(np_test, columns=test.columns)
		test_timestamps = test["timestamp"]
		test_data = test["value"]
		test_labels = test["target"]

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

training_slices = [slice(0, 2878)]
validation_slices = [slice(2878, 3580)]

match ALGORITHM:
	case "lstm":
		model = BraeiLSTM(window=30,
						  max_epochs=100,
						  batch_size=32,
						  batch_divide_training=True,
						  filename="lstm_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru":
		model = BraeiGRU(window=30,
						 max_epochs=100,
						 batch_size=32,
						 batch_divide_training=True,
						 filename="gru_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/gru_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn":
		model = BraeiGRU(window=30,
						 max_epochs=100,
						 batch_size=32,
						 batch_divide_training=True,
						 filename="cnn_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
			
	case "dense autoencoder":
		model = BraeiDenseAutoencoder(window=AUTOENCODER_WINDOW,
									  max_epochs=250,
									  batch_size=32,
									  filename="dense_ae_paper")
		
		if LOAD_MODEL:
			model.load_model("nn_models/dense_ae_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "lstm autoencoder":
		model = TimeSeriesAnomalyLSTMAutoencoder(window=AUTOENCODER_WINDOW,
												 max_epochs=100,
												 batch_size=32,
												 filename="lstm_ae_paper",
												 extend_not_multiple=True)
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_ae_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru autoencoder":
		model = BraeiGRU(window=AUTOENCODER_WINDOW,
						 max_epochs=100,
						 batch_size=32,
						 filename="gru_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/gru_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn autoencoder":
		model = BraeiGRU(window=AUTOENCODER_WINDOW,
						 max_epochs=100,
						 batch_size=32,
						 filename="cnn_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)

true_labels = data_test_labels
labels = model.predict(data, data_test)
scores = model.anomaly_score(data, data_test)

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	predictions = model.predict_time_series(data[validation_slices[0]], data_test)
	plot_time_series_forecast(data_test, predictions, on_same_plot=True)
	plot_time_series_forecast(data_test, predictions, on_same_plot=False)
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
