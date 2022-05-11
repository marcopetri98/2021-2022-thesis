import numpy as np
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "../data/dataset/"
DATASET = "nyc_taxi.csv"
PURE_DATA_KEY = "realKnownCause/nyc_taxi.csv"
GROUND_WINDOWS_PATH = "../data/dataset/combined_windows.json"

reader = NABTimeSeriesReader(DATASET_PATH)
all_df = reader.read(DATASET_PATH + DATASET).get_dataframe()

def preprocess(X) -> np.ndarray:
    return StandardScaler().fit_transform(X)

data_test = preprocess(np.array(all_df["value"]).reshape(all_df["value"].shape[0], 1))
data_test_labels = all_df["target"]
dataframe = all_df.copy()
dataframe["value"] = data_test