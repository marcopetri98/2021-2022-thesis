from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt.space import Categorical

from models.time_series.anomaly.statistical.TimeSeriesAnomalyARIMA import \
	TimeSeriesAnomalyARIMA
from models.time_series.anomaly.statistical.TimeSeriesAnomalySES import \
	TimeSeriesAnomalySES
from reader.NABTimeSeriesReader import NABTimeSeriesReader
from tuning.hyperparameter.TimeSeriesGridSearch import TimeSeriesGridSearch

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
DATASET_PATH = "data/dataset/"
DATASET = "nyc_taxi.csv"
PURE_DATA_KEY = "realKnownCause/nyc_taxi.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
TRAIN = True
LOAD_PREVIOUS = False
MODEL = "SES"
# kmeans, dbscan, lof, osvm, phase osvm, iforest, AR, MA, ARIMA, SES, ES

def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train=0.3).get_train_test_dataframes()

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
	# ARIMA models evaluation
	# model_ = TimeSeriesAnomalyARIMA(endog=train_data)
	# model_.set_params(**parameters)
	# results_ = model_.fit(method="statespace", gls=True, verbose=False, maxiter=2000)
	# return results_.bic
	
	# Exponential smoothing models evaluation
	model_ = TimeSeriesAnomalySES(ses_params={"endog": train_data})
	results_ = model_.fit(fit_params={"smoothing_level": parameters["alpha"]},
						  verbose=False)
	return results_.sse

hyper_searcher = TimeSeriesGridSearch([
										  # ARIMA parameters
										  # Categorical(create_ARIMA(0, 0, 30), name="order"),
										  # Categorical(["difference"], name="scoring"),
										  # Categorical([(0, 0, 0, 0)], name="seasonal_order"),
										  # Categorical(["n"], name="trend")
	
										  # ES parameters
										  Categorical([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], name="alpha")
									  ],
									  "data/searches/ses/",
									  "nyc_alpha",
									  TimeSeriesSplit(n_splits=5),
									  load_checkpoint=LOAD_PREVIOUS)
if TRAIN:
	hyper_searcher.search(data, data_labels, evaluate_time_series, verbose=True)

results = hyper_searcher.get_results()
results.print_search()

if MODEL == "ARIMA":
	model = 1
	orders = get_orders(model)
	scores = get_scores()
	if model == 1 or model == 0:
		orders_idx = np.argsort(np.array(orders))
		orders = np.array(orders)[orders_idx]
		scores = np.array(scores)[orders_idx]
		plot_single_search(orders, scores, title="AR/MA search")
	else:
		plot_ARMA_score(orders, scores)
else:
	history = results.get_history()
	alphas = [x[1] for x in history[1::]]
	scores = get_scores()
	plot_single_search(alphas, scores, title="SES alpha search")
