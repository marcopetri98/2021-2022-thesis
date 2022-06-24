import warnings
from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt.space import Categorical

from models.time_series.anomaly.statistical.TimeSeriesAnomalyARIMA import \
    TimeSeriesAnomalyARIMA
from reader.time_series.ODINTSReader import ODINTSReader
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
DATASET_NAME = "House20"
DATASET = DATASET_NAME + ".csv"
TRAIN = False
LOAD_PREVIOUS = True
MODEL = "ar"
TIMESTAMP_COL = "Time"
TIMESERIES_COL = "Appliance1"
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

print("Start to read dataset")
reader = ODINTSReader(DATASET_PATH + ANOMALIES_PREFIX + DATASET,
                      timestamp_col=TIMESTAMP_COL,
                      univariate_col=TIMESERIES_COL)
all_df = reader.read(DATASET_PATH + DATASET, resample=RESAMPLE).get_dataframe()
print("Dataset read")

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

warnings.filterwarnings("error")

def evaluate_time_series(train_data: np.ndarray,
                         train_labels: np.ndarray,
                         valid_data: np.ndarray,
                         valid_labels: np.ndarray,
                         parameters: dict) -> float:
    normal_data = np.argwhere(train_labels == 0)
    train_data = train_data[normal_data].reshape(-1, 1)
    
    # ARIMA models evaluation
    model_ = TimeSeriesAnomalyARIMA(endog=train_data)
    model_.set_params(**parameters)
    
    try:
        results_ = model_.fit(verbose=False)
        predictions = model_.anomaly_score(valid_data, train_data, verbose=False)
        return roc_auc_score(valid_labels.reshape(-1), predictions.reshape(-1))
    except Warning as w:
        print_warning("A warning ({}) has been raised from fitting or scoring."
                      " -1 is returned as score.".format(w.__class__))
        return -1
    except Exception as e:
        print_warning("An exception ({}) has been raised from fitting or scoring."
                      " -2 is returned as score.".format(e.__class__))
        return -2
    
    # Exponential smoothing models evaluation
    # model_ = TimeSeriesAnomalySES(ses_params={"endog": train_data})
    # results_ = model_.fit(fit_params={"smoothing_level": parameters["alpha"]},
    # 					  verbose=False)
    # return results_.sse

class MySplit(object):
    def __init__(self):
        super(MySplit, self).__init__()
    
    def split(self, proxy1, proxy2):
        return [(np.arange(0, TRAIN_END - TRAIN_VALID_START),
                 np.arange(TRAIN_END - TRAIN_VALID_START, TRAIN_END - TRAIN_VALID_START + TRAIN_VALID_END - VALID_START))]
    
    def get_n_splits(self):
        return 1

hyper_searcher = TimeSeriesGridSearch([
                                            # ARIMA parameters
                                            Categorical(create_ARIMA(5, [0, 1], 0), name="order"),
                                            Categorical(["difference"], name="scoring"),
                                            Categorical([(0, 0, 0, 0)], name="seasonal_order"),
                                            Categorical(["n"], name="trend")
                                            
                                            # ES parameters
                                            # Categorical([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], name="alpha")
                                      ],
                                      "data/searches/{}/".format(MODEL),
                                      DATASET_NAME,
                                      MySplit(),
                                      load_checkpoint=LOAD_PREVIOUS)
if TRAIN:
    hyper_searcher.search(data, data_labels, evaluate_time_series, verbose=True)

results = hyper_searcher.get_results()
results.print_search()

if MODEL == "arima" or MODEL == "ar" or MODEL == "ma":
    model = 0
    orders = get_orders(model)
    scores = get_scores()
    if model == 1 or model == 0:
        orders_idx = np.argsort(np.array(orders))
        orders = np.array(orders)[orders_idx]
        scores = np.array(scores)[orders_idx]
        plot_single_search(orders, scores, title="AR/MA search")
    else:
        plot_ARMA_score(orders, scores, wrong_val=0.56)
else:
    history = results.get_history()
    alphas = [x[1] for x in history[1::]]
    scores = get_scores()
    plot_single_search(alphas, scores, title="SES alpha search")
