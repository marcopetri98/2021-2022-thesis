from mleasy.reader.time_series.univariate import YahooS5Reader
from mleasy.visualizer import line_plot

BENCHMARK = "A2"
SERIES = 1

reader = YahooS5Reader("data/anomaly_detection/yahoo_s5/")
ds = reader.read(SERIES, benchmark=BENCHMARK).get_dataframe()
ds_opt = reader[67]

# for dataset in reader:
#     print(dataset.head())

print(ds.head())

line_plot(ds["timestamp"].values,
          ds["value"].values)

line_plot(ds_opt["timestamp"].values,
          ds_opt["value"].values)
