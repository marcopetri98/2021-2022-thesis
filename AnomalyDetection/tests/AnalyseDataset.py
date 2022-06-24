import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from analysis import TSDatasetAnalyser, StationarityTest

DATASET = "data/dataset/badec.csv"
MATPLOT_PRINT = True
USE_STL = False
LAG = 1
NLAGS = 20

# Exploratory data analysis
tmp_df = pd.read_csv(DATASET)
timestamps = tmp_df["ctime"]
values = tmp_df["device_consumption"]

diff = np.diff(values.values, 1)
diff = diff[1:]

plot_acf(diff, lags=NLAGS, alpha=0.0001)
plt.show()

dataset_analyser = TSDatasetAnalyser(values.values)

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
                                         difference_value=1)
