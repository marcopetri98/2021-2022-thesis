import time

import numpy as np
import pandas as pd
import skopt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Categorical, Real

from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN

#############################################
#											#
#											#
#				INSTANTIATORS				#
#											#
#											#
#############################################
FIRST_ROW = ["window",
			 #"stride",
			 #"score_method",
			 #"classification",
			 #"anomaly_threshold",
			 "eps",
			 "min_samples",
			 "performance"]
DBSCAN_SPACE = [
	Integer(2, 1000, name="window"),
	#Integer(1, 20, name="stride"),
	#Categorical(["z-score", "centroid"], name="score_method"),
	#Categorical(["voting", "points_score"], name="classification"),
	#Real(0.0, 1.0, name="anomaly_threshold"),
	Real(0.01, 20, name="eps"),
	Integer(2, 100, name="min_samples")
]

def dbscan_creator(**params):
	window = params["window"] if "window" in params.keys() else 200
	stride = params["stride"] if "stride" in params.keys() else 1
	score_method = params["score_method"] if "score_method" in params.keys() else "centroid"
	classification = params["classification"] if "classification" in params.keys() else "voting"
	anomaly_threshold = params["anomaly_threshold"] if "anomaly_threshold" in params.keys() else 0.0
	eps = params["eps"] if "eps" in params.keys() else 0.5
	min_samples = params["min_samples"] if "min_samples" in params.keys() else 5
	return TimeSeriesAnomalyDBSCAN(window=window,
								   stride=stride,
								   score_method=score_method,
								   classification=classification,
								   anomaly_threshold=anomaly_threshold,
								   eps=eps,
								   min_samples=min_samples)

#############################################
#											#
#											#
#		CONSTANTS AND PREPROCESSING			#
#											#
#											#
#############################################
DATASET = "nyc_taxi.csv"
DATASET_FOLDER = "dataset/"
TRAINING_PREFIX = "training_"
TESTING_PREFIX = "test_"
TRUTH_PREFIX = "truth"

ALGORITHM_SPACE = DBSCAN_SPACE
ESTIMATOR_CREATOR = dbscan_creator
MODEL_FOLDER = "dbscan"
CHECK_FILE = "taxi_window_eps_minsamples"
HAS_TO_LOAD_CHECKPOINT = False
HAS_TO_TRAIN = True
CALLS = 80
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

#############################################
#											#
#											#
#			ACTUAL OPTIMIZATION				#
#											#
#											#
#############################################
SEARCH_SPACE = ALGORITHM_SPACE

global counter
counter = 1

@skopt.utils.use_named_args(SEARCH_SPACE)
def objective(**params):
	estimator = ESTIMATOR_CREATOR(**params)
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

#############################################
#											#
#											#
#					MAIN					#
#											#
#											#
#############################################

if HAS_TO_TRAIN:
	start_time = time.time()
	
	res = _runSkoptOptimization()
	tries = [FIRST_ROW]
	for i in range(len(res.x_iters)):
		elem = res.x_iters[i].copy()
		elem.append(res.func_vals[i])
		tries.append(elem)
	tries = np.array(tries, dtype=object)
	np.save("searches/" + MODEL_FOLDER + "/" + CHECK_FILE, tries)
	
	print("\n\n\nTHE TRAINING PROCESS LASTED: ", time.time() - start_time)
else:
	tries = np.load("searches/" + MODEL_FOLDER + "/" + CHECK_FILE + ".npy",
					allow_pickle=True)
	indices = (np.argsort(tries[1:, -1]) + 1).tolist()
	indices.insert(0,0)
	tries = tries[np.array(indices)]
	
	print("Total number of tries: ", tries.shape[0] - 1)
	first = True
	for config in tries:
		if first:
			first = False
		else:
			text = ""
			for i in range(len(config)):
				text += str(tries[0, i])
				text += ": " + str(config[i]) + " "
			print(text)
