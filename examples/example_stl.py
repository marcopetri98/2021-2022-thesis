import matplotlib
import numpy as np

from anomalearn.analysis import decompose_time_series
from anomalearn.reader.time_series import ODINTSReader
from anomalearn.visualizer.time_series import plot_time_series_decomposition

matplotlib.rc("font", family="serif", serif=["Computer Modern Roman"], size=12)
matplotlib.rc("text", usetex=True)


DATASET_PATH = "../data/anomaly_detection/private_fridge/fridge1/"
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

trend, seasonal, residual = decompose_time_series(values, "stl", {"period": 115, "seasonal": 21})
plot_time_series_decomposition(values, seasonal, trend, residual, x_ticks_labels=ticks_labels, x_ticks_rotation=15, fig_size=(10, 8))
