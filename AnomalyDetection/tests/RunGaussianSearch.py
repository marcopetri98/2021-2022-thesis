import warnings
from datetime import datetime
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt.space import Categorical, Integer, Real

from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyIForest import \
    TimeSeriesAnomalyIForest
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyLOF import \
    TimeSeriesAnomalyLOF
from models.time_series.anomaly.machine_learning.TimeSeriesAnomalyOSVM import \
    TimeSeriesAnomalyOSVM
from models.time_series.anomaly.statistical.TimeSeriesAnomalyARIMA import \
    TimeSeriesAnomalyARIMA
from models.time_series.anomaly.statistical.TimeSeriesAnomalySES import \
    TimeSeriesAnomalySES
from reader.NABReader import NABReader
from reader.ODINTSReader import ODINTSReader
from tuning.hyperparameter.GaussianProcessesSearch import \
    GaussianProcessesSearch
from tuning.hyperparameter.TimeSeriesGridSearch import TimeSeriesGridSearch

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi

# ODIN TS
from utils.printing import print_warning

ANOMALIES_PREFIX = "anomalies_"

# NAB
PURE_DATA_KEY = "realKnownCause/nyc_taxi.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"

# SCRIPT
DATASET_PATH = "data/dataset/"
DATASET_NAME = "House11"
DATASET = DATASET_NAME + ".csv"
TRAIN = False
LOAD_PREVIOUS = True
MODEL = "lof"
TIMESTAMP_COL = "Time"
TIMESERIES_COL = "Appliance1"
N_CALLS = 40
N_INITIAL_POINTS = 3
RESAMPLE = True if DATASET_NAME in ["House1", "House11", "House20"] else False

if DATASET_NAME == "bae07" or DATASET_NAME == "badef":
    TRAIN_VALID_START = 0
    TRAIN_END = 40174
    VALID_START = 40174
    TRAIN_VALID_END = 44639
elif DATASET_NAME == "badec":
    TRAIN_VALID_START = 8640
    TRAIN_END = 48814
    VALID_START = 48814
    TRAIN_VALID_END = 53279
elif DATASET_NAME == "House1":
    TRAIN_VALID_START = 2054846 if not RESAMPLE else 999999999999
    TRAIN_END = 2385131 if not RESAMPLE else 999999999999
    VALID_START = 2385131 if not RESAMPLE else 999999999999
    TRAIN_VALID_END = 2421830 if not RESAMPLE else 999999999999
elif DATASET_NAME == "House11":
    TRAIN_VALID_START = 2131071 if not RESAMPLE else 234786
    TRAIN_END = 2463863 if not RESAMPLE else 273649
    VALID_START = 2463863 if not RESAMPLE else 273649
    TRAIN_VALID_END = 2500838 if not RESAMPLE else 277967
else:
    # House20.csv
    TRAIN_VALID_START = 7113 if not RESAMPLE else 719
    TRAIN_END = 368273 if not RESAMPLE else 40824
    VALID_START = 368273 if not RESAMPLE else 40824
    TRAIN_VALID_END = 408402 if not RESAMPLE else 45280

# kmeans, dbscan, lof, osvm, phase osvm, iforest, AR, MA, ARIMA, SES, ES

reader = ODINTSReader(DATASET_PATH + ANOMALIES_PREFIX + DATASET,
                      timestamp_col=TIMESTAMP_COL,
                      univariate_col=TIMESERIES_COL)
all_df = reader.read(DATASET_PATH + DATASET, resample=RESAMPLE).get_dataframe()

training = all_df.iloc[TRAIN_VALID_START:TRAIN_END]
training_validation = all_df.iloc[TRAIN_VALID_START:TRAIN_VALID_END]

scaler = StandardScaler()
scaler.fit(np.array(training["value"]).reshape((training["value"].shape[0], 1)))

data = scaler.transform(np.array(training_validation["value"]).reshape(training_validation["value"].shape[0], 1))
data_labels = training_validation["target"]
dataframe = training_validation.copy()
dataframe["value"] = data

# 0 = AR, 1 = MA, 2 = ARIMA
def get_orders(type: int = 0) -> list:
    history = results.get_history()
    if type == 0:
        orders = [x[1][0] for x in history[1::]]
    elif type == 1:
        orders = [x[1][2] for x in history[1::]]
    else:
        orders = [x[1] for x in history[1::]]
    return orders

def get_scores() -> list:
    history = results.get_history()
    scores = [x[0] for x in history[1::]]
    return scores


def plot_single_search(searched, score, fig_size: Tuple = (16, 6), title="Search"):
    fig = plt.figure(figsize=fig_size)
    plt.plot(searched,
             score,
             "b-",
             linewidth=0.5)
    plt.title(title)
    plt.show()
    
def create_ARIMA(ar: list | int, diff: list | int, ma: list | int) -> list:
    configs = []
    if isinstance(ar, int):
        ar = list(range(ar + 1))
    if isinstance(diff, int):
        diff = list(range(diff + 1))
    if isinstance(ma, int):
        ma = list(range(ma + 1))
    
    for ar_o in ar:
        for diff_o in diff:
            for ma_o in ma:
                configs.append((ar_o, diff_o, ma_o))
                
    return configs

def plot_ARMA_score(order, score, fig_ratio: float = 0.5, max_score: float = 1000.0, wrong_val: float = -1.0):
    matplotlib.use('Qt5Agg')
    fig = plt.figure(figsize=plt.figaspect(fig_ratio))
    ax = fig.add_subplot(projection='3d')
    x, y, z = [], [], []

    for i in range(len(order)):
        x.append(order[i][0])
        y.append(order[i][2])
        z.append(score[i])

    x, y, z = np.array(x), np.array(y), np.array(z)
    
    # replace error values with -1
    z[np.argwhere(z == -2)] = -1

    # eliminate points over the maximum score
    correct_points = np.argwhere(np.logical_and(z < max_score, z != -1))
    wrong_points = np.argwhere(np.logical_or(z >= max_score, z == -1))
    xw, yw, zw = x[wrong_points], y[wrong_points], z[wrong_points]
    zw[:] = wrong_val
    x, y, z = x[correct_points], y[correct_points], z[correct_points]
    
    x, y, z = x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))
    xw, yw, zw = xw.reshape((-1, 1)), yw.reshape((-1, 1)), zw.reshape((-1, 1))
    
    ax.scatter3D(x, y, z)
    ax.scatter3D(xw, yw, zw, c="r")
    ax.set_xlabel('AR order', fontweight='bold')
    ax.set_ylabel('MA order', fontweight='bold')
    ax.set_zlabel('AUC', fontweight='bold')
    plt.show()
    

def evaluate_time_series(train_data: np.ndarray,
                         train_labels: np.ndarray,
                         valid_data: np.ndarray,
                         valid_labels: np.ndarray,
                         parameters: dict) -> float:
    normal_data = np.argwhere(train_labels == 0)
    train_data = train_data[normal_data].reshape(-1, 1)
    train_labels = train_labels[normal_data].reshape(-1)
    
    warnings.filterwarnings("error")

    try:
        if MODEL == "ARIMA":
            model_ = TimeSeriesAnomalyARIMA(endog=train_data)
            parameters = dict(parameters)
            parameters["order"] = (
            parameters["ar_order"], parameters["diff_order"],
            parameters["ma_order"])
            del parameters["ar_order"]
            del parameters["diff_order"]
            del parameters["ma_order"]
            model_.set_params(**parameters)
            results_ = model_.fit(verbose=False)
            predictions = model_.anomaly_score(valid_data, train_data, verbose=False)
        elif MODEL == "iforest":
            model_ = TimeSeriesAnomalyIForest(parameters["window"],
                                              n_estimators=parameters["n_estimators"],
                                              max_samples=parameters["max_samples"],
                                              random_state=22)
            model_.fit(train_data, train_labels)
            predictions = model_.anomaly_score(valid_data)
        elif MODEL == "lof":
            model_ = TimeSeriesAnomalyLOF(parameters["window"],
                                          novelty=True,
                                          n_neighbors=parameters["n_neighbors"],
                                          scaling="none")
            model_.fit(train_data, train_labels)
            predictions = model_.anomaly_score(valid_data)
        else:
            model_ = TimeSeriesAnomalyOSVM(parameters["window"],
                                           gamma=parameters["gamma"],
                                           tol=parameters["tol"],
                                           nu=parameters["nu"])
            model_.fit(train_data, train_labels)
            predictions = model_.anomaly_score(valid_data)
            
        warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return 1 - roc_auc_score(valid_labels.reshape(-1), predictions.reshape(-1))
    except Warning as w:
        print_warning("A warning ({}) has been raised from fitting or scoring."
                      " 1 is returned as score.".format(w.__class__))
        warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return 1
    except Exception as e:
        print_warning("An exception ({}) has been raised from fitting or scoring."
                      " 2 is returned as score.".format(e.__class__))
        warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return 2


class MySplit(object):
    def __init__(self):
        super(MySplit, self).__init__()
        
    def split(self, proxy1, proxy2):
        return [(np.arange(0, TRAIN_END - TRAIN_VALID_START),
                 np.arange(TRAIN_END - TRAIN_VALID_START, TRAIN_END - TRAIN_VALID_START + TRAIN_VALID_END - VALID_START))]
    
    def get_n_splits(self):
        return 1

hyper_searcher = GaussianProcessesSearch([
                                            # ARIMA parameters
                                            # Integer(0, 5, name="ar_order"),
                                            # Categorical([1], name="diff_order"),
                                            # Integer(0, 5, name="ma_order"),
                                            # Categorical(["difference"], name="scoring"),
                                            # Categorical(["n"], name="trend")
    
                                            # Isolation forest parameters
                                            # Integer(1, 30, name="window"),
                                            # Integer(20, 200, name="n_estimators"),
                                            # Integer(150, 400, name="max_samples")
    
                                            # LOF parameters
                                            Integer(1, 40, name="window"),
                                            Integer(1, 300, name="n_neighbors")
    
                                            # OSVM parameters
                                            # Integer(1, 40, name="window"),
                                            # Real(0.001, 1, name="gamma"),
                                            # Real(1e-10, 0.1, name="tol", prior="log-uniform"),
                                            # Real(0.001, 0.5, name="nu")
                                         ],
                                         "data/searches/{}/".format(MODEL.lower()),
                                         DATASET_NAME + "_gp",
                                         MySplit(),
                                         load_checkpoint=LOAD_PREVIOUS,
                                         gp_kwargs={"n_calls": N_CALLS, "n_initial_points": N_INITIAL_POINTS})
if TRAIN:
    hyper_searcher.search(data, data_labels, evaluate_time_series, verbose=True)

results = hyper_searcher.get_results()
results.print_search()

if MODEL == "ARIMA":
    model = 2
    orders = get_orders(model)
    scores = get_scores()
    if model == 1 or model == 0:
        orders_idx = np.argsort(np.array(orders))
        orders = np.array(orders)[orders_idx]
        scores = np.array(scores)[orders_idx]
        plot_single_search(orders, scores, title="AR/MA search")
    else:
        plot_ARMA_score(orders, scores)
elif MODEL == "SES":
    history = results.get_history()
    alphas = [x[1] for x in history[1::]]
    scores = get_scores()
    plot_single_search(alphas, scores, title="SES alpha search")
