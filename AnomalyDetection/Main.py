import numpy as np
import pandas as pd

from models.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
import visualizer.Viewer as vw

ALGORITHM = "dbscan"

model = None
params = None
anomalies = None
anomalies_score = None
tmp_df = pd.read_csv("dataset/truth_ambient_temperature_system_failure.csv")
timestamps = tmp_df["timestamp"]
temperature = tmp_df["value"]

data_original = np.array(temperature)
data_original = data_original.reshape(data_original.shape[0], 1)
data_stationary = np.array(temperature.diff(1))
data_stationary = data_stationary[1:]
data_stationary = data_stationary.reshape(data_stationary.shape[0], 1)

match ALGORITHM:
	case "k-means":
		pass
	
	case "dbscan":
		model = TimeSeriesAnomalyDBSCAN(0.5, 5)
		model.fit(data_stationary, 200, 200)
		anomalies = model.get_anomalies()
		anomalies_score = model.get_anomaly_scores()
		anomalies = np.append(np.array([0]), anomalies)
		anomalies_score = np.append(np.array([0]), anomalies_score)
	
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

vw.plot_univariate_time_series(tmp_df)
vw.plot_univariate_time_series_predictions(tmp_df,
										   anomalies_score)
