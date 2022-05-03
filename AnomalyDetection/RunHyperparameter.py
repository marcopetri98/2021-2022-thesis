from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Categorical

from models.time_series.anomaly.TimeSeriesAnomalyARIMA import \
	TimeSeriesAnomalyARIMA
from reader.NABTimeSeriesReader import NABTimeSeriesReader

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
from tuning.hyperparameter.TimeSeriesGridSearch import TimeSeriesGridSearch

DATASET_PATH = "data/dataset/"
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
TRAIN = False
LOAD_PREVIOUS = True

def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train_perc=0.3).get_train_test_dataframes()

data = preprocess(np.array(training["value"]).reshape(training["value"].shape[0], 1))
data_labels = training["target"]
dataframe = training.copy()
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


def plot_AR_MA_score(order, score, fig_size: Tuple = (16, 6)):
	fig = plt.figure(figsize=fig_size)
	plt.plot(order,
			 score,
			 "b-",
			 linewidth=0.5)
	plt.title("AR/MA search")
	plt.show()
	
def create_ARIMA(ar: list, diff: list, ma: list) -> list:
	configs = []
	
	for ar_o in ar:
		for diff_o in diff:
			for ma_o in ma:
				configs.append((ar_o, diff_o, ma_o))
				
	return configs

def plot_ARMA_score(order, score, fig_ratio: float = 0.5, max_score: float = 1000.0):
	matplotlib.use('Qt5Agg')
	fig = plt.figure(figsize=plt.figaspect(fig_ratio))
	ax = fig.add_subplot(projection='3d')
	x, y, z = [], [], []

	for i in range(len(order)):
		x.append(order[i][0])
		y.append(order[i][2])
		z.append(score[i])

	x, y, z = np.array(x), np.array(y), np.array(z)
	
	# eliminate points over the maximum score
	correct_points = np.argwhere(z < max_score)
	wrong_points = np.argwhere(z >= max_score)
	xw, yw, zw = x[wrong_points], y[wrong_points], z[wrong_points]
	zw[:] = -1
	x, y, z = x[correct_points], y[correct_points], z[correct_points]
	
	x, y, z = x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))
	xw, yw, zw = xw.reshape((-1, 1)), yw.reshape((-1, 1)), zw.reshape((-1, 1))
	
	ax.scatter3D(x, y, z)
	ax.scatter3D(xw, yw, zw, c="r")
	ax.set_xlabel('AR order', fontweight='bold')
	ax.set_ylabel('MA order', fontweight='bold')
	ax.set_zlabel('BIC', fontweight='bold')
	plt.show()

def evaluate_time_series(train_data: np.ndarray,
						 train_labels: np.ndarray,
						 valid_data: np.ndarray,
						 valid_labels: np.ndarray,
						 parameters: dict) -> float:
	model = TimeSeriesAnomalyARIMA(endog=train_data)
	model.set_params(**parameters)
	results = model.fit(method="statespace", gls=True, verbose=False, maxiter=2000)
	return results.bic

hyper_searcher = TimeSeriesGridSearch([
										  #Integer(2, 100, name="window"),
										  # Integer(1, 20, name="stride"),
										  # Real(0.0, 1.0, name="anomaly_threshold"),
										  # Real(0.00001, 0.5, prior="log-uniform", name="contamination"),
										  #Integer(2, 100, name="n_estimators"),
										  Categorical(create_ARIMA([1, 5, 10, 15, 20, 25, 30, 35, 40],
																   [1],
																   [1, 5, 10, 15, 20, 25, 30, 35, 40]), name="order")
									  ],
									  "data/searches/arima/",
									  "temp_no_refit_gls_statespace_arima",
									  TimeSeriesSplit(n_splits=5),
									  load_checkpoint=LOAD_PREVIOUS)
if TRAIN:
	hyper_searcher.search(data, data_labels, evaluate_time_series, verbose=True)

results = hyper_searcher.get_results()
results.print_search()

model = 2
orders = get_orders(model)
scores = get_scores()
if model == 1 or model == 0:
	orders_idx = np.argsort(np.array(orders))
	orders = np.array(orders)[orders_idx]
	scores = np.array(scores)[orders_idx]
	plot_AR_MA_score(orders, scores)
else:
	plot_ARMA_score(orders, scores)
