import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from get_windows_indices import get_windows_indices
from models.time_series.anomaly.deep_learning.BraeiCNN import BraeiCNN
from models.time_series.anomaly.deep_learning.BraeiCNNBatch import BraeiCNNBatch
from models.time_series.anomaly.deep_learning.BraeiDenseAutoencoder import \
	BraeiDenseAutoencoder
from models.time_series.anomaly.deep_learning.BraeiGRU import BraeiGRU
from models.time_series.anomaly.deep_learning.BraeiLSTM import BraeiLSTM
from models.time_series.anomaly.deep_learning.CNNAutoencoder import \
	CNNAutoencoder
from models.time_series.anomaly.deep_learning.GRUAutoencoder import \
	GRUAutoencoder
from models.time_series.anomaly.deep_learning.LSTMAutoencoder import LSTMAutoencoder
from reader.NABTimeSeriesReader import NABTimeSeriesReader
from visualizer.Viewer import plot_time_series_forecast, \
	plot_time_series_with_predicitons_bars

ALGORITHM = "lstm autoencoder"

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
VALIDATION_DIM = 0.2
DATASET_PATH = "dataset/"
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "dataset/combined_windows.json"
ALL_METRICS = True
LOAD_MODEL = False
CHECK_OVERFITTING = False
AUTOENCODER = True
AUTOENCODER_WINDOW = 32


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

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train_perc=0.5).get_train_test_dataframes()

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
		
		np_test = np.array(test[remainder_points:])
		test = pd.DataFrame(np_test, columns=test.columns)

# Data used to evaluate
dataframe = test.copy()
dataframe["value"] = data_test

if CHECK_OVERFITTING:
	data_test = preprocess(np.array(training["value"]).reshape(training["value"].shape[0], 1))
	data_test_labels = training["target"]
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
validation_slices = [slice(2878, 3582)]

match ALGORITHM:
	case "lstm":
		model = BraeiLSTM(window=30,
						  max_epochs=100,
						  batch_size=32,
						  batch_divide_training=True,
						  distribution="truncated_gaussian",
						  filename="lstm_paper_tg")
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_paper_tg")
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
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = BraeiCNN(window=30,
						 max_epochs=100,
						 batch_size=32,
						 batch_divide_training=True,
						 filename="cnn_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn-batch":
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = BraeiCNNBatch(window=30,
							  max_epochs=100,
							  batch_size=32,
							  batch_divide_training=True,
							  filename="cnnb_paper")
		if LOAD_MODEL:
			model.load_model("nn_models/cnnb_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
			
	case "dense autoencoder":
		model = BraeiDenseAutoencoder(window=AUTOENCODER_WINDOW,
									  max_epochs=250,
									  batch_size=32,
									  perc_quantile=0.98,
									  filename="dense_ae_paper_ov")
		
		if LOAD_MODEL:
			model.load_model("nn_models/dense_ae_paper_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "lstm autoencoder":
		model = LSTMAutoencoder(window=AUTOENCODER_WINDOW,
								max_epochs=500,
								batch_size=32,
								folder_save_path="nn_models/custom/",
								filename="autoencoder_lstm_1",
								#distribution="truncated_gaussian",
								extend_not_multiple=True,
								test_overlapping=True)
		if LOAD_MODEL:
			model.load_model("nn_models/lstm_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru autoencoder":
		model = GRUAutoencoder(window=AUTOENCODER_WINDOW,
							   max_epochs=500,
							   batch_size=32,
							   filename="gru_ae_ov",
							   extend_not_multiple=True)
		if LOAD_MODEL:
			model.load_model("nn_models/gru_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn autoencoder":
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = CNNAutoencoder(window=AUTOENCODER_WINDOW,
							   max_epochs=500,
							   batch_size=32,
							   filename="cnn_ae_ov",
							   extend_not_multiple=True)
		if LOAD_MODEL:
			model.load_model("nn_models/cnn_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)

true_labels = np.asarray(data_test_labels, dtype=np.intc)
labels = model.predict(data, data_test)
scores = model.anomaly_score(data, data_test)
perc = np.sum(labels) / labels.shape[0]

if ALL_METRICS:
	print("MODEL BEST VALIDATION ERROR: %f" % model.validation_best_error_)
	compute_metrics(true_labels, scores, labels, compute_roc_auc=not CHECK_OVERFITTING, only_roc_auc=False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	predictions = model.predict_time_series(data[validation_slices[0]], data_test)
	plot_time_series_forecast(data_test, predictions, on_same_plot=True)
	plot_time_series_forecast(data_test, predictions, on_same_plot=False)
	
	bars = get_windows_indices(all_df,
							   PURE_DATA_KEY,
							   GROUND_WINDOWS_PATH)
	all_timestamps = all_df["timestamp"].tolist()
	bars = [dataframe["timestamp"].tolist().index(all_timestamps[int(bar)])
			for bar in bars
			if all_timestamps[int(bar)] in dataframe["timestamp"].tolist()]
	bars = np.array(bars)
	plot_time_series_with_predicitons_bars(dataframe,
										   labels,
										   bars,
										   pred_color='r')
else:
	compute_metrics(true_labels, scores, compute_roc_auc=not CHECK_OVERFITTING, only_roc_auc=True)
