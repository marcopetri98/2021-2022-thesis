import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.metrics import precision_recall_curve

from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.anomaly.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF
import visualizer.Viewer as vw
from models.anomaly.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM

ALGORITHM = "one class svm"
UNSUPERVISED = False

params = None
if UNSUPERVISED:
	tmp_df = pd.read_csv("dataset/truth_ambient_temperature_system_failure.csv")
	timestamps = tmp_df["timestamp"]
	training_data = tmp_df["value"]
	training_labels = tmp_df["target"]
else:
	tmp_df = pd.read_csv("dataset/training_ambient_temperature_system_failure.csv")
	timestamps = tmp_df["timestamp"]
	training_data = tmp_df["value"]
	training_labels = tmp_df["target"]
	
	tmp_df_test = pd.read_csv("dataset/test_ambient_temperature_system_failure.csv")
	testing_data = np.array(tmp_df_test["value"])
	testing_labels = np.array(tmp_df_test["target"])
	testing_data = testing_data.reshape((testing_data.shape[0], 1))
	testing_labels = testing_labels.reshape((testing_labels.shape[0], 1))

training_labels = np.array(training_labels)
data_original = np.array(training_data)
data_original = data_original.reshape(data_original.shape[0], 1)
data_stationary = np.array(training_data.diff(1))
data_stationary = data_stationary[1:]
data_stationary = data_stationary.reshape(data_stationary.shape[0], 1)

data = data_original

scaler = preprocessing.StandardScaler()
scaler.fit(data)
data_prep = scaler.transform(data)

data = data_prep

match ALGORITHM:
	case "k-means":
		pass
	
	case "dbscan":
		model = TimeSeriesAnomalyDBSCAN(5,
										20,
										window=168,
										stride=1,
										score_method="centroid",
										classification="point_threshold",
										anomaly_threshold=0.6)
		model.fit(data)
	
	case "hdbscan":
		pass
	
	case "isolation forest":
		pass
	
	case "one class svm":
		model = TimeSeriesAnomalyOSVM(window=672,
									  stride=1)
		model.fit(data, training_labels)
	
	case "local outlier factor":
		model = TimeSeriesAnomalyLOF(window=17,
									 stride=5,
									 classification="auto",
									 anomaly_threshold=0.0)
		model.fit(data)
	
	case "lstm":
		pass
	
	case "gru":
		pass
	
	case "cnn":
		pass
	
	case "lstm autoencoder":
		pass
	
	case "gru autoencoder":
		pass
	
	case "cnn autoencoder":
		pass

if UNSUPERVISED:
	anomalies = model.get_anomalies()
	metrics_labels = training_labels
	plot_df = tmp_df
else:
	anomalies = model.predict(testing_data)
	metrics_labels = testing_labels
	plot_df = tmp_df_test

anomalies_score = model.get_anomaly_scores()
#anomalies = np.append(np.array([0]), anomalies)
#anomalies_score = np.append(np.array([0]), anomalies_score)

confusion_matrix = metrics.confusion_matrix(metrics_labels, anomalies)
precision = metrics.precision_score(metrics_labels, anomalies)
recall = metrics.recall_score(metrics_labels, anomalies)
f1_score = metrics.f1_score(metrics_labels, anomalies)
accuracy = metrics.accuracy_score(metrics_labels, anomalies)
pre, rec, _ = precision_recall_curve(metrics_labels, anomalies_score, pos_label=1)
auc = metrics.auc(rec, pre)

print("ACCURACY SCORE: ", accuracy)
print("PRECISION SCORE: ", precision)
print("RECALL SCORE: ", recall)
print("F1 SCORE: ", f1_score)
print("AUC SCORE: ", auc)

vw.plot_roc_curve(metrics_labels, anomalies_score)
vw.plot_precision_recall_curve(metrics_labels, anomalies_score)
vw.plot_confusion_matrix(confusion_matrix)
vw.plot_time_series_ndarray(data)
vw.plot_univariate_time_series(plot_df)
vw.plot_univariate_time_series_predictions(plot_df, anomalies)
vw.plot_univariate_time_series_predictions(plot_df, anomalies_score)
