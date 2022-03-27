import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Real

from tuning.hyperparameter.HyperparameterSearch import HyperparameterSearch
from models.anomaly.TimeSeriesAnomalyDBSCAN import TimeSeriesAnomalyDBSCAN

DATASET = "ambient_temperature_system_failure.csv"
DATASET_PATH = "dataset/"
TRAINING_PATH = DATASET_PATH + "training/"

training = pd.read_csv(TRAINING_PATH + DATASET)
training_timestamps = training["timestamp"]
training_data = training["value"]
training_labels = training["target"]

data = StandardScaler().fit_transform(np.array(training_data).reshape(training_data.shape[0], 1))
data_labels = training_labels
dataframe = training.copy()
dataframe["value"] = data

hyper_searcher = HyperparameterSearch(TimeSeriesAnomalyDBSCAN(),
									  [
										  Integer(1, 20, name="window"),
										  #Integer(1, 20, name="stride"),
										  #Categorical(["z-score", "centroid"], name="score_method"),
										  #Categorical(["voting", "points_score"], name="classification"),
										  #Real(0.0, 1.0, name="anomaly_threshold"),
										  Real(0.001, 5, name="eps"),
										  Integer(2, 100, name="min_samples")
									  ],
									  "searches/dbscan/",
									  "temperature_window_eps_minsamples",
									  200,
									  10,
									  train_and_test=False,
									  load_checkpoint=False)
hyper_searcher.fit(data, data_labels)
hyper_searcher.print_search()
