import glob
import os
from datetime import datetime, timedelta

import pandas as pd

from analysis import TSDatasetAnalyser, StationarityTest

MATPLOT_PRINT = True
USE_STL = False
LAG = 1
NLAGS = 100

#
# Thanks for the support to
# Start Author: Nicolò Oreste Pinciroli Vago
# Politecnico di Milano: https://www.deib.polimi.it/eng/people/details/1116006
#
def keep_cycles(df, threshold=50):
    df["variation"] = df["device_consumption"].diff()
    # see where there is the beginning of a new cycle
    mask = df["variation"] > threshold
    beginnings = df[mask].index.values
    if len(beginnings) > 0:
        # index at which the first cycle begins
        first_beginning = beginnings[0]
        # index at which the last cycle ends
        last_beginning = beginnings[-1] - 1
        
        if first_beginning < last_beginning:
            return df[first_beginning:last_beginning]


dataset_name = "badef"

dfs = [None] * len(glob.glob("data/CERTH/" + dataset_name + "/train/clean_data/*.csv"))

for filename in glob.glob("data/CERTH/" + dataset_name + "/train/clean_data/*.csv"):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        file_number = int(filename.split("clean_" + dataset_name + "_")[1].split(".csv")[0])
        df = pd.read_csv(f)
        df = keep_cycles(df)
        dfs[file_number] = df

periodic_timeseries = pd.concat(dfs)

start_datetime = datetime(2020, 1, 1, 0, 0)
end_datetime = start_datetime + timedelta(minutes=len(periodic_timeseries) - 1)

timerange = pd.date_range(start=start_datetime, end=end_datetime, freq="1min")
periodic_timeseries["ctime"] = timerange
periodic_timeseries.reset_index(inplace=True)
#
# Thanks for the support to
# End Author: Nicolò Oreste Pinciroli Vago
# Politecnico di Milano: https://www.deib.polimi.it/eng/people/details/1116006
#

values = periodic_timeseries["device_consumption"]

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
                                         difference_value=1,
                                         fig_size=(21,12))
