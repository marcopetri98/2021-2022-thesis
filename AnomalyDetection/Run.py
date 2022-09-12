import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from mleasy import visualizer as vw
from mleasy.models.time_series.anomaly.machine_learning.TSAKMeans import \
	TSAKMeans
from mleasy.models.time_series.anomaly.statistical.TSAARIMA import TSAARIMA
from mleasy.models.time_series.anomaly import TimeSeriesAnomalyDBSCAN
from mleasy.models.time_series.anomaly.statistical.TSAES import TSAES
from mleasy.models.time_series.anomaly.machine_learning.TSAIsolationForest import TSAIsolationForest
from mleasy.models.time_series.anomaly.machine_learning.TSALOF import TSALOF
from mleasy.models.time_series.anomaly import TimeSeriesAnomalyOSVM
from mleasy.models.time_series.anomaly import TimeSeriesAnomalyOSVMPhase
from mleasy.models.time_series.anomaly.statistical.TSASES import TSASES
from mleasy.reader.MissingStrategy import MissingStrategy

#################################
#								#
#								#
#		RUN CONSTANTS			#
#								#
#								#
#################################
from mleasy.reader.time_series.ODINTSReader import ODINTSReader

ALGORITHM = "AR"

# ODIN TS
ANOMALIES_PREFIX = "anomalies_"
THRESHOLD = 11
# np.max(scores.reshape(-1))
# np.min(scores.reshape(-1))
MIN_THRESHOLD = 9
MAX_THRESHOLD = 23

# kmeans, dbscan, lof, osvm, phase osvm, iforest, AR, MA, ARIMA, SES, ES
# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
DATASET_PATH = "data/dataset/"
DATASET = "House1.csv"
PHASE = "testing"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
WHERE_TO_SAVE = "{}_{}_{}".format(ALGORITHM, PHASE ,DATASET[:-4])
ALL_METRICS = True
CHECK_OVERFITTING = False
ALL_DATA = False
UNSUPERVISED = False
SELF_SUPERVISED = True
SAVE_RESULTS = True
RESAMPLE = True if DATASET in ["House1.csv", "House11.csv", "House20.csv"] else False
DROPPED_MISSING = False

if PHASE == "validation":
	if DATASET == "bae07.csv" or DATASET == "badef.csv":
		TRAIN_START = 0
		TRAIN_END = 40174
		TEST_START = 40174
		TEST_END = 44639
	elif DATASET == "badec.csv":
		TRAIN_START = 8640
		TRAIN_END = 48814
		TEST_START = 48814
		TEST_END = 53279
	elif DATASET == "House1.csv":
		TRAIN_START = 2054846 if not RESAMPLE else 227338 if DROPPED_MISSING else 292974
		TRAIN_END = 2385131 if not RESAMPLE else 267370 if DROPPED_MISSING else 333150
		TEST_START = 2385131 if not RESAMPLE else 267370 if DROPPED_MISSING else 333150
		TEST_END = 2421830 if not RESAMPLE else 271818 if DROPPED_MISSING else 337614
	elif DATASET == "House11.csv":
		TRAIN_START = 2131071 if not RESAMPLE else 234786 if DROPPED_MISSING else 259944
		TRAIN_END = 2463863 if not RESAMPLE else 273649 if DROPPED_MISSING else 300120
		TEST_START = 2463863 if not RESAMPLE else 273649 if DROPPED_MISSING else 300120
		TEST_END = 2500838 if not RESAMPLE else 277967 if DROPPED_MISSING else 304584
	else:
		# House20.csv
		TRAIN_START = 7113 if not RESAMPLE else 719 if DROPPED_MISSING else 719
		TRAIN_END = 368273 if not RESAMPLE else 40824 if DROPPED_MISSING else 40895
		TEST_START = 368273 if not RESAMPLE else 40824 if DROPPED_MISSING else 40895
		TEST_END = 408402 if not RESAMPLE else 45280 if DROPPED_MISSING else 45359
else:
	if DATASET == "bae07.csv" or DATASET == "badef.csv":
		TRAIN_START = 0
		TRAIN_END = 44639
		TEST_START = 48960
		TEST_END = 90719
	elif DATASET == "badec.csv":
		TRAIN_START = 8640
		TRAIN_END = 53279
		TEST_START = 57600
		TEST_END = 99359
	elif DATASET == "House1.csv":
		TRAIN_START = 2054846 if not RESAMPLE else 227338 if DROPPED_MISSING else 292974
		TRAIN_END = 2421830 if not RESAMPLE else 271818 if DROPPED_MISSING else 337614
		TEST_START = 2804097 if not RESAMPLE else 314886 if DROPPED_MISSING else 380814
		TEST_END = 3133655 if not RESAMPLE else 358656 if DROPPED_MISSING else 425454
	elif DATASET == "House11.csv":
		TRAIN_START = 2131071 if not RESAMPLE else 234786 if DROPPED_MISSING else 259944
		TRAIN_END = 2500838 if not RESAMPLE else 277967 if DROPPED_MISSING else 304584
		TEST_START = 2885051 if not RESAMPLE else 322581 if DROPPED_MISSING else 349224
		TEST_END = 3233333 if not RESAMPLE else 362888 if DROPPED_MISSING else 389544
	else:
		# House20.csv
		TRAIN_START = 7113 if not RESAMPLE else 719 if DROPPED_MISSING else 719
		TRAIN_END = 408402 if not RESAMPLE else 45280 if DROPPED_MISSING else 45359
		TEST_START = 745915 if not RESAMPLE else 88469 if DROPPED_MISSING else 88559
		TEST_END = 1116986 if not RESAMPLE else 133102 if DROPPED_MISSING else 133199

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

reader = ODINTSReader(DATASET_PATH + ANOMALIES_PREFIX + DATASET,
					  timestamp_col="Time",
					  univariate_col="Appliance1")
all_df = reader.read(DATASET_PATH + DATASET,
					 resample=RESAMPLE,
					 missing_strategy=MissingStrategy.FIXED_VALUE).get_dataframe()
#training, test = reader.train_test_split(train=0.37).get_train_test_dataframes()
training = all_df.iloc[TRAIN_START:TRAIN_END]
test = all_df.iloc[TEST_START:TEST_END]

normal_data = np.argwhere(training["target"].values == 0)
training = training.iloc[normal_data.reshape(-1)]

#################################
#								#
#								#
#			DEFINE DATA			#
#								#
#								#
#################################
scaler = StandardScaler()
scaler.fit(np.array(training["value"]).reshape((training["value"].shape[0], 1)))
print("Mean is {} and std is {}".format(scaler.mean_, scaler.scale_))

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
	model = TSAKMeans(window=3,
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
	model = TSALOF(window=3,
				   classification="points_score",
				   anomaly_portion=0.01,
				   n_neighbors=100)
elif SELF_SUPERVISED and ALGORITHM == "lof":
	# Better window=3, neighbors=40
	model = TSALOF(window=17,
				   classification="points_score",
				   scaling="none",
				   n_neighbors=21,
				   novelty=True)
	model.fit(train)
	model.threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "osvm":
	model = TimeSeriesAnomalyOSVM(window=25,
								  gamma=0.001,
								  nu=0.5,
								  tol=1e-10,
								  classification="points_score",
								  anomaly_portion=0.0015)
	model.fit(train)
	model.threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "phase osvm":
	model = TimeSeriesAnomalyOSVMPhase(windows=[5, 10, 20],
									   nu=0.4,
									   classification="points_score",
									   anomaly_portion=0.0015)
	model.fit(train)
elif SELF_SUPERVISED and ALGORITHM == "iforest":
	model = TSAIsolationForest(window=29,
                               n_estimators=20,
                               max_samples=210,
                               #contamination=0.0004,
                               classification="points_score",
                               random_state=22)
	model.fit(train)
	model.threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "AR":
	model = TSAARIMA(scoring="difference",
                     endog=train,
                     order=(1, 0, 0),
                     seasonal_order=(0, 0, 0, 0),
                     trend="n")
	model.fit()
	model._threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "MA":
	model = TSAARIMA(scoring="difference",
                     endog=train,
                     order=(0, 0, 8),
                     seasonal_order=(0, 0, 0, 0),
                     trend="n")
	model.fit(fit_params={"method":"innovations_mle",
						  "gls":True})
elif SELF_SUPERVISED and ALGORITHM == "ARIMA":
	model = TSAARIMA(endog=train,
                     order=(1, 1, 5),
                     seasonal_order=(0, 0, 0, 0),
                     perc_quantile=0.98)
	model.fit()
	model._threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "SES":
	model = TSASES(ses_params={"endog": train})
	model.fit(fit_params={"optimized": True,
						  "use_brute": True})
elif SELF_SUPERVISED and ALGORITHM == "ES":
	model = TSAES(es_params={"endog": train,
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

if PHASE == "validation":
	if ALGORITHM in ["iforest", "lof", "osvm", "AR", "ARIMA"]:
		print("Evaluate threshold at different values: ")
	
	for value in np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, 21, dtype=np.double):
		if ALGORITHM in ["iforest", "lof", "osvm"]:
			model.threshold = value
			varying_labels = model.classify(data_test.reshape((-1, 1)), verbose=False)
			f1_score = metrics.f1_score(true_labels, varying_labels)
			print("Threshold {:.5f} has F1 {}".format(value, f1_score))
		elif ALGORITHM in ["AR", "ARIMA"]:
			model._threshold = value
			varying_labels = model.classify(data_test.reshape((-1, 1)), data.reshape((-1, 1)), verbose=False)
			f1_score = metrics.f1_score(true_labels, varying_labels)
			print("Threshold {:.5f} has F1 {}".format(value, f1_score))
			
	if ALGORITHM in ["iforest", "lof", "osvm", "AR", "ARIMA"]:
		print("\n\n\n")

if ALL_METRICS:
	compute_metrics(true_labels, scores, labels, only_roc_auc=False)
	make_metric_plots(dataframe, true_labels, scores, labels)
	
	if ALGORITHM in ["ARIMA", "AR", "MA", "SES", "ES"]:
		predictions = model.predict_time_series(data, data_test)
		vw.plot_time_series_forecast(data_test, predictions, on_same_plot=True)
		vw.plot_time_series_forecast(data_test, predictions, on_same_plot=False)
	
	# bars = vw.get_bars_indices_on_test_df(all_df,
	# 									  dataframe,
	# 									  PURE_DATA_KEY,
	# 									  GROUND_WINDOWS_PATH)
	# vw.plot_time_series_with_predicitons_bars(dataframe,
	# 										  labels,
	# 										  bars,
	# 										  pred_color='r')
else:
	compute_metrics(true_labels, scores, only_roc_auc=True)
