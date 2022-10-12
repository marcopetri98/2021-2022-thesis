import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from Metrics import compute_metrics, make_metric_plots
from mleasy import visualizer as vw
from mleasy.models.time_series.anomaly.machine_learning.TSAKMeans import \
    TSAKMeans
from mleasy.models.time_series.anomaly.statistical.TSAARIMA import TSAARIMA
from mleasy.models.time_series.anomaly.machine_learning import TSADBSCAN
from mleasy.models.time_series.anomaly.statistical.TSAES import TSAES
from mleasy.models.time_series.anomaly.machine_learning.TSAIsolationForest import TSAIsolationForest
from mleasy.models.time_series.anomaly.machine_learning.TSALOF import TSALOF
from mleasy.models.time_series.anomaly.machine_learning import TSAOCSVM
from mleasy.models.time_series.anomaly.machine_learning import TSAOCSVMPhase
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

ALGORITHM = "MA"

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
DATASET_PATH = "data/anomaly_detection/private_fridge/fridge1/"
DATASET = "fridge1.csv"
PHASE = "validation"
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
                      timestamp_col="ctime",
                      univariate_col="device_consumption")
all_df = reader.read(DATASET_PATH + "all_" + DATASET,
                     resample=RESAMPLE,
                     missing_strategy=MissingStrategy.FIXED_VALUE).get_dataframe()

if PHASE == "validation":
    training, test = reader.train_valid_test_split(train=0.6, valid=0.2).get_train_test_dataframes()
else:
    training, test = reader.train_test_split(train=0.6).get_train_test_dataframes()

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
    model = TSADBSCAN(window=3,
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
    model = TSAOCSVM(window=25,
                     gamma=0.001,
                     nu=0.5,
                     tol=1e-10,
                     classification="points_score",
                     anomaly_portion=0.0015)
    model.fit(train)
    model.threshold = THRESHOLD
elif SELF_SUPERVISED and ALGORITHM == "phase osvm":
    model = TSAOCSVMPhase(windows=[5, 10, 20],
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
