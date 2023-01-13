import matplotlib.pyplot as plt
import numpy as np

from mleasy.analysis import TSDatasetAnalyser, DecompositionMethod
from mleasy.reader.time_series import ODINTSReader

DATASET_PATH = "data/anomaly_detection/private_fridge/fridge1/"
DATASET = "fridge1.csv"
ANOMALIES_PREFIX = "anomalies_"
START = 2000
INCREMENT = 2000

reader = ODINTSReader(DATASET_PATH + ANOMALIES_PREFIX + DATASET,
					  timestamp_col="ctime",
					  univariate_col="device_consumption")
all_df = reader.read(DATASET_PATH + "all_" + DATASET).get_dataframe()
values = all_df["value"].values
values = values[START:START + INCREMENT]
ticks_labels_idx = np.linspace(START, START + INCREMENT - 1, 10, dtype=np.int64)
ticks_labels = all_df["timestamp"].values
ticks_labels = ticks_labels[ticks_labels_idx]

dataset_analyser = TSDatasetAnalyser(values)

plt.rcParams.update({"font.size": 16, "font.family": "serif"})
dataset_analyser.decompose_time_series(DecompositionMethod.STL,
                                       method_params={"period": 115, "seasonal": 21},
									   x_ticks_labels=ticks_labels,
									   x_ticks_rotation=15)
