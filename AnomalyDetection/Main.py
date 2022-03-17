import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics

from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
import visualizer.Viewer as vw

ALGORITHM = "dbscan"

model = None
params = None
anomalies = None
anomalies_score = None
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

data = data_original

match ALGORITHM:
	case "k-means":
		pass
	
	case "dbscan":
		model = TimeSeriesAnomalyDBSCAN(0.5,
										50,
										window=1440,
										stride=168,
										anomaly_threshold=0.6,
										use_score=True,
										score_method="centroid")
		model.fit(data)
		anomalies = model.get_anomalies()
		anomalies_score = model.get_anomaly_scores()
		#anomalies = np.append(np.array([0]), anomalies)
		#anomalies_score = np.append(np.array([0]), anomalies_score)
	
	case "hdbscan":
		pass
	
	case "isolation forest":
		pass
	
	case "one class svm":
		pass
	
	case "local outlier factor":
		pass
	
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
