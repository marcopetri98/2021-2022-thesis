import numpy as np
import pandas as pd

from reader.MissingStrategy import MissingStrategy
from reader.ODINTSReader import ODINTSReader
from utils.json import save_py_json

DATASET = "House20"
TIMESTAMP = "Time"
SERIES = "Appliance1"
odin_reader = ODINTSReader("data/dataset/anomalies_{}.csv".format(DATASET),
                           TIMESTAMP,
                           SERIES,
									 "start_date",
									 "end_date")
odin_reader.read("data/dataset/{}.csv".format(DATASET), resample=True, missing_strategy=MissingStrategy.FIXED_VALUE)
df = odin_reader.get_dataframe()

# badec: train: 2020-01-21 00:00:00 to 2020-02-20 23:59:00 test 2020-02-24 00:00:00 to 2020-03-23 23:59:00
# badef: train: 2020-01-21 00:00:00 to 2020-02-20 23:59:00 test 2020-02-24 00:00:00 to 2020-03-23 23:59:00
# bae07: train: 2020-01-21 00:00:00 to 2020-02-20 23:59:00 test 2020-02-24 00:00:00 to 2020-03-23 23:59:00
# House1: train: 2014-05-01 00:00:00 to 2014-05-31 23:59:00 test 2014-07-01 00:00:00 to 2014-07-31 23:59:00
# House11: train: 2014-12-01 00:00:00 to 2014-12-31 23:59:00 test 2015-02-01 00:00:00 to 2015-02-28 23:59:00
# House20: train: 2014-03-21 00:00:00 to 2014-04-20 23:59:00 test 2014-05-21 00:00:00 to 2014-06-20 23:59:00

train_valid_start = np.argwhere(df["timestamp"].values == "2014-03-21 00:00:00")[0,0]
train_valid_end = np.argwhere(df["timestamp"].values == "2014-04-20 23:59:00")[0,0]
train_end_valid_start = int(train_valid_end - (train_valid_end - train_valid_start)*0.1 + 1)
test_start = np.argwhere(df["timestamp"].values == "2014-05-21 00:00:00")[0,0]
test_end = np.argwhere(df["timestamp"].values == "2014-06-20 23:59:00")[0,0]
print("train-valid start %s" % train_valid_start)
print("train-valid end %s" % (train_valid_end + 1))
print("train end / valid start %s" % train_end_valid_start)

print("test start %s" % test_start)
print("test end %s" % (test_end + 1))

validation_slice = slice(train_end_valid_start, train_valid_end + 1)
testing_slice = slice(test_start, test_end + 1)

validation = df[validation_slice].copy()
testing = df[testing_slice].copy()

# enhanced_dataset = odin_reader.get_complete_dataframe()
# enhanced_dataset = enhanced_dataset.drop(columns=["value", "target"])

# validation_prop = enhanced_dataset[validation_slice].copy()
# testing_prop = enhanced_dataset[testing_slice].copy()
# validation_prop.to_csv("validation_{}_properties.csv".format(DATASET), index=False)
# testing_prop.to_csv("testing_{}_properties.csv".format(DATASET), index=False)

# print(np.sum(validation["target"].values))
#
validation_anomalies = { "anomalies" : [] }
testing_anomalies = { "anomalies" : [] }
for index, data in validation.iterrows():
	if data["target"] == 1:
		validation_anomalies["anomalies"].append(data["timestamp"])
for index, data in testing.iterrows():
	if data["target"] == 1:
		testing_anomalies["anomalies"].append(data["timestamp"])

validation = validation.drop(columns=["target"])
testing = testing.drop(columns=["target"])

df.to_csv("resampled_{}_gt.csv".format(DATASET), index=False)
save_py_json(validation_anomalies, "validation_{}_anomalies.json".format(DATASET))
validation.to_csv("validation_{}_gt.csv".format(DATASET), index=False)
save_py_json(testing_anomalies, "testing_{}_anomalies.json".format(DATASET))
testing.to_csv("testing_{}_gt.csv".format(DATASET), index=False)

HAND_START = 48961 - 1
HAND_END = 90720 - 1

print(df.iloc[HAND_START - 9:HAND_START + 1])
print(df.iloc[HAND_END:HAND_END + 10])
