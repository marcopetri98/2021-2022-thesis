from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Categorical

from models.time_series.anomaly import TimeSeriesAnomalyLOF
from models.time_series.anomaly.TimeSeriesAnomalyARIMA import \
	TimeSeriesAnomalyARIMA
from reader.NABTimeSeriesReader import NABTimeSeriesReader
from tuning.hyperparameter.GaussianProcessesSearch import GaussianProcessesSearch

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
from tuning.hyperparameter.TimeSeriesGridSearch import TimeSeriesGridSearch

DATASET_PATH = "data/dataset/"
DATASET = "ambient_temperature_system_failure.csv"
PURE_DATA_KEY = "realKnownCause/ambient_temperature_system_failure.csv"
GROUND_WINDOWS_PATH = "data/dataset/combined_windows.json"
TRAIN = True
LOAD_PREVIOUS = False

def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train_perc=0.3).get_train_test_dataframes()

data = preprocess(np.array(training["value"]).reshape(training["value"].shape[0], 1))
data_labels = training["target"]
dataframe = training.copy()
dataframe["value"] = data

# 0 = AR, 1 = MA, 2 = ARMA
def get_orders(type: int = 0) -> list:
	history = results.get_history()
	if type == 0:
		orders = [x[1][0] for x in history[1::]]
	else:
		orders = [x[1][2] for x in history[1::]]
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
	
def plot_ARMA_score(order, score, fig_ratio):
	pass

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
										  Categorical([(0, 1, 1),
													   (0, 1, 5),
													   (0, 1, 10),
													   (0, 1, 15),
													   (0, 1, 20),
													   (0, 1, 25),
													   (0, 1, 30),
													   (0, 1, 35),
													   (0, 1, 40)], name="order")
									  ],
									  "data/searches/ma/",
									  "no_refit_gls_statespace_ma",
									  TimeSeriesSplit(n_splits=5),
									  load_checkpoint=LOAD_PREVIOUS)
if TRAIN:
	hyper_searcher.search(data, data_labels, evaluate_time_series, verbose=True)

results = hyper_searcher.get_results()
results.print_search()

orders = get_orders(1)
scores = get_scores()
orders_idx = np.argsort(np.array(orders))
orders = np.array(orders)[orders_idx]
scores = np.array(scores)[orders_idx]
plot_AR_MA_score(orders, scores)
