import numpy as np
import pandas as pd
import skopt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Categorical, Real

from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN

DATASET = "nyc_taxi.csv"
DATASET_FOLDER = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
TRUTH_PREFIX = "truth"

MODEL_FOLDER = "dbscan"
CHECK_FILE = "19_03_2022"
HAS_TO_LOAD_CHECKPOINT = False
CALLS = 100
INITIAL_STARTS = 10

def preprocess(X) -> np.ndarray:
	return StandardScaler().fit_transform(X)

training = pd.read_csv(DATASET_FOLDER + TRAINING_PREFIX + DATASET)
training_timestamps = training["timestamp"]
training_data = training["value"]
training_labels = training["target"]

data = preprocess(np.array(training_data).reshape(training_data.shape[0], 1))
data_labels = training_labels
dataframe = training.copy()
dataframe["value"] = data

SEARCH_SPACE = [
	Integer(2, 1000, name="window"),
	#Integer(1, 20, name="stride"),
	Categorical(["z-score", "centroid"], name="score_method"),
	Categorical(["voting", "points_score"], name="classification"),
	Real(0.0, 1.0, name="anomaly_threshold"),
	Real(0.01, 20, name="eps"),
	Integer(2, 100, name="min_samples")
]

global counter
counter = 1

@skopt.utils.use_named_args(SEARCH_SPACE)
def objective(**params):
	window = params["window"]
	score_method = params["score_method"]
	classification = params["classification"]
	anomaly_threshold = params["anomaly_threshold"]
	eps = params["eps"]
	min_samples = params["min_samples"]
	estimator = TimeSeriesAnomalyDBSCAN(window=window,
										stride=1,
										score_method=score_method,
										classification=classification,
										anomaly_threshold=anomaly_threshold,
										eps=eps,
										min_samples=min_samples)
	global counter
	print("Runnin ", counter, " k-fold")
	print("\tRun params: ", params)
	counter += 1
	
	k_fold_score = 0
	cross_val_gen = KFold(n_splits=5)
	
	for train, test in cross_val_gen.split(data, data_labels):
		try:
			y_pred = estimator.fit_predict(data[test])
			k_fold_score += metrics.f1_score(data_labels[test], y_pred, zero_division=0)
		except ValueError:
			pass
	k_fold_score = k_fold_score / cross_val_gen.get_n_splits()
	
	return 1 - k_fold_score

def _runSkoptOptimization():
	checkpoint_saver = CheckpointSaver("searches/" + MODEL_FOLDER + "/" + CHECK_FILE + ".pkl", compress=9)
	
	if HAS_TO_LOAD_CHECKPOINT:
		previous_checkpoint = skopt.load("searches/" + MODEL_FOLDER + "/" + CHECK_FILE + ".pkl")
		x0 = previous_checkpoint.x_iters
		y0 = previous_checkpoint.func_vals
		results = skopt.gp_minimize(objective,
									SEARCH_SPACE,
									x0=x0,
									y0=y0,
									n_calls=CALLS,
									n_initial_points=INITIAL_STARTS,
									callback=[checkpoint_saver])
	else:
		results = skopt.gp_minimize(objective,
									SEARCH_SPACE,
									n_calls=CALLS,
									n_initial_points=INITIAL_STARTS,
									callback=[checkpoint_saver])
	
	return results

res = _runSkoptOptimization()
tries = [["window", "score_method", "classification", "anomaly_threshold", "eps", "min_samples", 1000]]
for i in range(len(res.x_iters)):
	elem = res.x_iters[i].copy()
	elem.append(res.func_vals[i])
	tries.append(elem)
tries = np.array(tries, dtype=object)
np.save("searches/" + MODEL_FOLDER + "/" + CHECK_FILE, tries)
