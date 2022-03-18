import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics

from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.anomaly.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF
import visualizer.Viewer as vw
from models.anomaly.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM

ALGORITHM = "one class svm"

params = None
tmp_df = pd.read_csv("dataset/truth_ambient_temperature_system_failure.csv")
timestamps = tmp_df["timestamp"]
temperature = tmp_df["value"]
true_labels = tmp_df["target"]

data_original = np.array(temperature)
data_original = data_original.reshape(data_original.shape[0], 1)
data_stationary = np.array(temperature.diff(1))
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
		model = TimeSeriesAnomalyOSVM(window=24,
									  stride=1,
									  classification="auto",
									  anomaly_threshold=1)
		model.fit(data)
	
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


anomalies = model.get_anomalies()
anomalies_score = model.get_anomaly_scores()
#anomalies = np.append(np.array([0]), anomalies)
#anomalies_score = np.append(np.array([0]), anomalies_score)

confusion_matrix = metrics.confusion_matrix(true_labels, anomalies)
precision = metrics.precision_score(true_labels, anomalies)
recall = metrics.recall_score(true_labels, anomalies)
f1_score = metrics.f1_score(true_labels, anomalies)
accuracy = metrics.accuracy_score(true_labels, anomalies)

print("ACCURACY SCORE: ", accuracy)
print("PRECISION SCORE: ", precision)
print("RECALL SCORE: ", recall)
print("F1 SCORE: ", f1_score)

vw.plot_roc_curve(true_labels, anomalies_score)
vw.plot_precision_recall_curve(true_labels, anomalies_score)
vw.plot_confusion_matrix(confusion_matrix)
vw.plot_time_series_ndarray(data)
vw.plot_univariate_time_series(tmp_df)
vw.plot_univariate_time_series_predictions(tmp_df, anomalies)
vw.plot_univariate_time_series_predictions(tmp_df, anomalies_score)
