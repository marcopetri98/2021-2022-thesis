import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
import visualizer.Viewer as vw
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyKMeans import \
	TimeSeriesAnomalyKMeans
from models.time_series.anomaly.statistical.TimeSeriesAnomalyARIMA import TimeSeriesAnomalyARIMA
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN
from models.time_series.anomaly.statistical.TimeSeriesAnomalyES import TimeSeriesAnomalyES
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyIForest import TimeSeriesAnomalyIForest
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyLOF import TimeSeriesAnomalyLOF
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyOSVM import TimeSeriesAnomalyOSVM
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyOSVMPhase import TimeSeriesAnomalyOSVMPhase
from models.time_series.anomaly.statistical.TimeSeriesAnomalySES import TimeSeriesAnomalySES
from reader.NABTimeSeriesReader import NABTimeSeriesReader

#################################
#								#
#								#
#		RUN CONSTANTS			#
#								#
#								#
#################################

ALGORITHM = "iforest"

# kmeans, dbscan, lof, osvm, phase osvm, iforest, AR, MA, ARIMA, SES, ES
# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
DATASET_PATH = "data/dataset/"
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
WHERE_TO_SAVE = ALGORITHM + "_" + DATASET[:-4] + "_predictions"
ALL_METRICS = True
CHECK_OVERFITTING = False
ALL_DATA = False
UNSUPERVISED = False
SELF_SUPERVISED = True
SAVE_RESULTS = True


def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

#################################
#								#
#								#
#			LOAD DATA			#
#								#
#								#
#################################
model = None

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train=0.37).get_train_test_dataframes()

#################################
#								#
#								#
#			DEFINE DATA			#
#								#
#								#
#################################
scaler = StandardScaler()
scaler.fit(np.array(training["value"]).reshape((training["value"].shape[0], 1)))

# Data used to train
data = scaler.transform(np.array(training["value"]).reshape((training["value"].shape[0], 1)))
data_labels = training["target"]

# Data used to test
data_test = scaler.transform(np.array(test["value"]).reshape((test["value"].shape[0], 1)))
data_test_labels = test["target"]

# Dataframe used to evaluate
dataframe = test.copy()
dataframe["value"] = data_test

train = data

if CHECK_OVERFITTING:
	data_test = data
	data_test_labels = data_labels
	dataframe = training.copy()
	dataframe["value"] = data_test
elif ALL_DATA:
	data_test = preprocess(np.array(all_df["value"]).reshape(all_df["value"].shape[0], 1))
	data_test_labels = all_df["target"]
	dataframe = all_df.copy()
	dataframe["value"] = data_test

#################################
#								#
#								#
#			FIT MODEL			#
#								#
#								#
#################################
if UNSUPERVISED and ALGORITHM == "kmeans":
	model = TimeSeriesAnomalyKMeans(window=3,
									classification="points_score",
									anomaly_portion=0.0003,
									anomaly_threshold=0.9888,
									kmeans_params={"n_clusters": 4,
												   "random_state": 22})
elif UNSUPERVISED and ALGORITHM == "dbscan":
	model = TimeSeriesAnomalyDBSCAN(window=3,
									eps=1.0,
									min_samples=150,
									#anomaly_threshold=0.9888,
									anomaly_portion=0.0003,
									classification="voting")
elif UNSUPERVISED and ALGORITHM == "lof":
	# Better window=3, neighbors=40
	model = TimeSeriesAnomalyLOF(window=3,
								 classification="points_score",
								 anomaly_portion=0.01,
								 n_neighbors=100)
elif SELF_SUPERVISED and ALGORITHM == "lof":
	# Better window=3, neighbors=40
	model = TimeSeriesAnomalyLOF(window=3,
								 classification="points_score",
								 n_neighbors=40,
								 novelty=True)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "osvm":
	model = TimeSeriesAnomalyOSVM(window=10,
								  nu=0.4,
								  classification="points_score",
								  anomaly_portion=0.0015)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "phase osvm":
	model = TimeSeriesAnomalyOSVMPhase(windows=[5, 10, 20],
									   nu=0.4,
									   classification="points_score",
									   anomaly_portion=0.0015)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "iforest":
	model = TimeSeriesAnomalyIForest(window=10,
									 n_estimators=100,
									 contamination=0.0004,
									 # max_samples=30,
									 random_state=22)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "AR":
	model = TimeSeriesAnomalyARIMA(scoring="difference",
								   endog=train,
								   order=(10, 0, 0),
								   seasonal_order=(0, 0, 0, 0),
								   trend="n")
	model.fit(fit_params={"method":"innovations_mle",
						  "gls":True})
elif SELF_SUPERVISED and ALGORITHM == "MA":
	model = TimeSeriesAnomalyARIMA(scoring="difference",
								   endog=train,
								   order=(0, 0, 8),
								   seasonal_order=(0, 0, 0, 0),
								   trend="n")
	model.fit(fit_params={"method":"innovations_mle",
						  "gls":True})
elif SELF_SUPERVISED and ALGORITHM == "ARIMA":
	model = TimeSeriesAnomalyARIMA(endog=train, order=(1, 1, 2), perc_quantile=0.98)
	model.fit(fit_params={"method":"statespace",
						  "gls":True})
elif SELF_SUPERVISED and ALGORITHM == "SES":
	model = TimeSeriesAnomalySES(ses_params={"endog": train})
	model.fit(fit_params={"optimized": True,
						  "use_brute": True})
elif SELF_SUPERVISED and ALGORITHM == "ES":
	model = TimeSeriesAnomalyES(es_params={"endog": train,
										   "trend": "add",
										   "damped_trend": True,
										   "seasonal": "add",
										   "seasonal_periods": 7})
	model.fit(fit_params={"optimized": True})

#################################
#								#
#								#
#			EVALUATE			#
#								#
#								#
#################################
true_labels = data_test_labels
if ALGORITHM in ["ARIMA", "AR", "MA"]:
	labels = model.classify(data_test.reshape((-1, 1)), data.reshape((-1, 1)))
	scores = model.anomaly_score(data_test.reshape((-1, 1)), data.reshape((-1, 1)))
else:
	labels = model.classify(data_test.reshape((-1, 1)))
	scores = model.anomaly_score(data_test.reshape((-1, 1)))

if SAVE_RESULTS:
	new_frame: pd.DataFrame = test.copy()
	new_frame = new_frame.drop(columns=["target", "value"])
	
	labels_frame: pd.DataFrame = new_frame.copy()
	scores_frame: pd.DataFrame = new_frame.copy()
	
	labels_frame.insert(len(labels_frame.columns), "confidence", labels)
	scores_frame.insert(len(scores_frame.columns), "confidence", scores)
	
	labels_frame.to_csv(WHERE_TO_SAVE + "_labels.csv", index=False)
	scores_frame.to_csv(WHERE_TO_SAVE + "_scores.csv", index=False)
	
	if ALGORITHM in ["ARIMA", "AR", "MA", "SES", "ES"]:
		# compute predictions
		regressions = model.predict_time_series(train, data_test)
		regressions = (regressions * scaler.scale_) + scaler.mean_
		regressions_frame: pd.DataFrame = new_frame.copy()
		regressions_frame.insert(len(regressions_frame.columns), "value", regressions)
		regressions_frame.to_csv(WHERE_TO_SAVE + "_regressions.csv", index=False)

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, only_roc_auc=False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	
	if ALGORITHM in ["ARIMA", "AR", "MA", "SES", "ES"]:
		predictions = model.predict_time_series(data, data_test)
		vw.plot_time_series_forecast(data_test, predictions, on_same_plot=True)
		vw.plot_time_series_forecast(data_test, predictions, on_same_plot=False)
	
	bars = vw.get_bars_indices_on_test_df(all_df,
										  dataframe,
										  PURE_DATA_KEY,
										  GROUND_WINDOWS_PATH)
	vw.plot_time_series_with_predicitons_bars(dataframe,
											  labels,
											  bars,
											  pred_color='r')
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
