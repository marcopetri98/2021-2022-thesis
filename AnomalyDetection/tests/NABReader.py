from reader.time_series.NABReader import NABReader

DATASET_PATH = "data/dataset/"
DATASET = "ambient_temperature_system_failure.csv"
WHERE_TO_SAVE = "ambient_gt_labels.csv"

reader = NABReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()
training, test = reader.train_test_split(train=0.37).get_train_test_dataframes()

print(test.head())

modified_df = test.drop(columns=["value"])
modified_df.to_csv(WHERE_TO_SAVE, index=False)