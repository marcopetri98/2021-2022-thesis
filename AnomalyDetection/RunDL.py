import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from mleasy.models.time_series.anomaly import BraeiCNN
from mleasy.models.time_series.anomaly import BraeiCNNBatch
from mleasy.models.time_series.anomaly.deep_learning.BraeiDenseAutoencoder import BraeiDenseAutoencoder
from mleasy.models.time_series.anomaly import BraeiGRU
from mleasy.models.time_series.anomaly.deep_learning.BraeiLSTM import BraeiLSTM
from mleasy.models.time_series.anomaly import CNNAutoencoder
from mleasy.models.time_series.anomaly.deep_learning.GRUAutoencoder import GRUAutoencoder
from mleasy.models.time_series.anomaly.deep_learning.LSTMAutoencoder import LSTMAutoencoder
from mleasy.reader.time_series.univariate import NABReader
from mleasy.visualizer.Viewer import plot_time_series_forecast, plot_time_series_with_predicitons_bars, get_bars_indices_on_test_df

ALGORITHM = "cnn"

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
VALIDATION_DIM = 0.2
DATASET_PATH = "data/dataset/"
DATASET = "nyc_taxi.csv"
PURE_DATA_KEY = "realKnownCause/nyc_taxi.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
# The number of normal samples that must lie between a normal slice and an anomaly
SAFETY_DIM = 50
MINIMUM_SLICE = 100
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

reader = NABReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = train_test_split(all_df, train_size=0.37, shuffle=False)

#################################
#								#
#								#
#			DEFINE DATA			#
#								#
#								#
#################################
# Data used to train
data = preprocess(np.array(training["value"]).reshape((training["value"].shape[0], 1)))
data_labels = training["target"]

# Data used to test
data_test = preprocess(np.array(test["value"]).reshape((test["value"].shape[0], 1)))
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
change_idx = np.where(np.array(data_labels[:-1]) != np.array(data_labels[1:]))
change_idx = np.array(change_idx) + 1
change_idx = change_idx.reshape(change_idx.shape[1])
change_idx = np.concatenate((np.array([0]), change_idx))
normal_slices = []
anomaly_slices = []
for i in range(len(change_idx)):
	start = change_idx[i]
	if i < len(change_idx) - 1:
		stop = change_idx[i + 1]
	else:
		stop = data.shape[0]

	if data_labels[change_idx[i]] == 1:
		anomaly_slices.append(slice(start, stop))
	else:
		normal_slices.append(slice(start, stop))

print("The slices before safety check: %s" % normal_slices)

for i in range(len(normal_slices)):
	start_first = False
	end_last = False
	if normal_slices[i].start == 0:
		start_first = True
	if normal_slices[i].stop == data.shape[0]:
		end_last = True

	if not start_first and not end_last:
		normal_slices[i] = slice(normal_slices[i].start + SAFETY_DIM,
								 normal_slices[i].stop - SAFETY_DIM,
								 normal_slices[i].step)
	elif not start_first and end_last:
		normal_slices[i] = slice(normal_slices[i].start + SAFETY_DIM,
								 normal_slices[i].stop,
								 normal_slices[i].step)
	elif start_first and not end_last:
		normal_slices[i] = slice(normal_slices[i].start,
								 normal_slices[i].stop - SAFETY_DIM,
								 normal_slices[i].step)

print("The slices after safety check: %s" % normal_slices)

# Gets the step (equal for all slices)
step = normal_slices[0].step if normal_slices[0].step is not None else 1
ok_normal_slices = []
tot_samples = 0
for i in range(len(normal_slices)):
	samples = int((normal_slices[i].stop - normal_slices[i].start) / step)
	if samples > MINIMUM_SLICE:
		tot_samples += samples
		ok_normal_slices.append(normal_slices[i])

valid_points = tot_samples * VALIDATION_DIM
training_slices = []
validation_slices = []
tot_valid = 0
for i in reversed(range(len(ok_normal_slices))):
	if tot_valid >= valid_points:
		training_slices.append(ok_normal_slices[i])
	else:
		samples = int(
			(ok_normal_slices[i].stop - ok_normal_slices[i].start) / step)
		needed_points = valid_points - tot_valid

		if samples < needed_points:
			validation_slices.append(ok_normal_slices[i])
		else:
			train_stop = int(ok_normal_slices[i].stop - needed_points * step)
			slice_train = slice(ok_normal_slices[i].start,
								train_stop,
								ok_normal_slices[i].step)
			slice_valid = slice(train_stop,
								ok_normal_slices[i].stop,
								ok_normal_slices[i].step)

			# If points are enough, they are added to lists
			if (slice_train.stop - slice_train.start) / step >= MINIMUM_SLICE:
				training_slices.append(slice_train)
			if (slice_valid.stop - slice_valid.start) / step >= MINIMUM_SLICE:
				validation_slices.append(slice_valid)

			# Since in this else we enter only at the last slice, set up to exit in case minimum
			# slice is not reached
			tot_valid = valid_points + 1

training_slices.reverse()
validation_slices.reverse()

print("The training slices are: %s" % training_slices)
print("The validation slices are: %s" % validation_slices)

match ALGORITHM:
	case "lstm":
		model = BraeiLSTM(window=30,
						  max_epochs=100,
						  batch_size=32,
						  predict_validation=0.1982,
						  batch_divide_training=True,
						  distribution="truncated_gaussian",
						  filename="lstm_paper_tg")
		if LOAD_MODEL:
			model.load_model("data/nn_models/lstm_paper_tg")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru":
		model = BraeiGRU(window=30,
						 max_epochs=100,
						 batch_size=32,
						 predict_validation=0.1982,
						 batch_divide_training=True,
						 filename="gru_paper")
		if LOAD_MODEL:
			model.load_model("data/nn_models/gru_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn":
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = BraeiCNN(window=30,
						 max_epochs=50,
						 batch_size=32,
						 predict_validation=0.1982,
						 batch_divide_training=True,
						 filename="nyc_cnn_paper")
		if LOAD_MODEL:
			model.load_model("data/nn_models/nyc_cnn_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn-batch":
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = BraeiCNNBatch(window=30,
							  max_epochs=100,
							  batch_size=32,
							  predict_validation=0.1982,
							  batch_divide_training=True,
							  filename="cnnb_paper")
		if LOAD_MODEL:
			model.load_model("data/nn_models/cnnb_paper")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
			
	case "dense autoencoder":
		model = BraeiDenseAutoencoder(window=AUTOENCODER_WINDOW,
									  max_epochs=50,
									  batch_size=32,
									  predict_validation=0.1982,
									  distribution="mahalanobis",
									  perc_quantile=0.98,
									  filename="nyc_braei_dense_gauss",
									  test_overlapping=False)
		
		if LOAD_MODEL:
			model.load_model("data/nn_models/nyc_braei_dense_gauss")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "lstm autoencoder":
		model = LSTMAutoencoder(window=AUTOENCODER_WINDOW,
                                max_epochs=500,
                                batch_size=32,
                                folder_save_path="data/nn_models/custom/",
                                #distribution="truncated_gaussian",
                                extend_not_multiple=True,
                                test_overlapping=True,
                                filename="autoencoder_lstm_1")
		if LOAD_MODEL:
			model.load_model("data/nn_models/lstm_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "gru autoencoder":
		model = GRUAutoencoder(window=AUTOENCODER_WINDOW,
							   max_epochs=500,
							   batch_size=32,
							   extend_not_multiple=True,
							   filename="gru_ae_ov")
		if LOAD_MODEL:
			model.load_model("data/nn_models/gru_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)
	
	case "cnn autoencoder":
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		model = CNNAutoencoder(window=AUTOENCODER_WINDOW,
							   max_epochs=500,
							   batch_size=32,
							   extend_not_multiple=True,
							   filename="cnn_ae_ov")
		if LOAD_MODEL:
			model.load_model("data/nn_models/cnn_ae_ov")
		else:
			model.fit(data, training_slices, validation_slices, data_labels)

true_labels = np.asarray(data_test_labels, dtype=np.intc)
labels = model.predict(data, data_test)
scores = model.anomaly_score(data, data_test)
perc = np.sum(labels) / labels.shape[0]

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, compute_roc_auc=not CHECK_OVERFITTING, only_roc_auc=False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	predictions = model.predict_time_series(data[validation_slices[0]], data_test)
	plot_time_series_forecast(data_test, predictions, on_same_plot=True)
	plot_time_series_forecast(data_test, predictions, on_same_plot=False)
	
	bars = get_bars_indices_on_test_df(all_df,
										  dataframe,
										  PURE_DATA_KEY,
										  GROUND_WINDOWS_PATH)
	plot_time_series_with_predicitons_bars(dataframe,
										   labels,
										   bars,
										   pred_color='r')
else:
	compute_metrics(true_labels, scores, compute_roc_auc=not CHECK_OVERFITTING, only_roc_auc=True)
