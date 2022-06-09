import pandas as pd

from reader.NABTimeSeriesReader import NABTimeSeriesReader

# DATASET 1: ambient_temperature_system_failure
# DATASET 2: nyc_taxi
DATASET_PATH = "data/dataset/"
DATASET = "ambient_temperature_system_failure.csv"
WHERE_TO_SAVE = "ambient_gt_labels.csv"

#################################
#								#
#								#
#			LOAD DATA			#
#								#
#								#
#################################
model = None

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()

print(all_df.head())

modified_df = all_df.drop(columns=["value"])
modified_df.to_csv(WHERE_TO_SAVE)