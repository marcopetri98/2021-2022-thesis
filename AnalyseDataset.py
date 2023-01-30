import numpy as np

from anomalearn.analysis import TSDatasetAnalyser, StationarityTest, DecompositionMethod
from anomalearn.reader.time_series import ODINTSReader
from anomalearn.reader import MissingStrategy

MATPLOT_PRINT = True
USE_STL = False
LAG = 1
NLAGS = 10

DATASET_PATH = "data/anomaly_detection/private_fridge/fridge1/"
DATASET = "fridge1.csv"
ANOMALIES_PREFIX = "anomalies_"
RESAMPLE = False
START = 38
INCREMENT = 4000

reader = ODINTSReader(DATASET_PATH + ANOMALIES_PREFIX + DATASET,
					  timestamp_col="ctime",
					  univariate_col="device_consumption")
all_df = reader.read(DATASET_PATH + "all_" + DATASET,
					 resample=RESAMPLE,
					 missing_strategy=MissingStrategy.FIXED_VALUE).get_dataframe()
values = all_df["value"].values
values = values[START:START + INCREMENT]
ticks_labels_idx = np.linspace(START, START + INCREMENT - 1, 10, dtype=np.int64)
ticks_labels = all_df["timestamp"].values
ticks_labels = ticks_labels[ticks_labels_idx]

dataset_analyser = TSDatasetAnalyser(values)

dataset_analyser.analyse_stationarity(StationarityTest.ADFULLER)

dataset_analyser.analyse_stationarity(StationarityTest.KPSS)

dataset_analyser.analyse_stationarity(StationarityTest.ADFULLER,
                                      difference_series=True,
                                      difference_value=1)

dataset_analyser.analyse_stationarity(StationarityTest.KPSS,
                                      difference_series=True,
                                      difference_value=1)

dataset_analyser.show_acf_pacf_functions({"nlags": NLAGS, "alpha": 0.0001},
                                         {"nlags": NLAGS, "alpha": 0.0001},
                                         difference_series=True,
                                         difference_value=1,
                                         fig_size=(21,12))

dataset_analyser.decompose_time_series(DecompositionMethod.STL,
									   # STL PARAMS
                                        method_params={"period": 115,
													   "seasonal": 21,
													   "robust": True},
                                       # MOVING AVERAGE PARAMS
                                       # method_params={"period": 115, "model": "additive"},
									   #x_ticks_labels=ticks_labels,
									   x_ticks_rotation=15)
