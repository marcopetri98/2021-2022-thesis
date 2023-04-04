import matplotlib
import numpy as np

from anomalearn.reader.time_series import UCRReader
from anomalearn.visualizer import line_plot

matplotlib.rc("font", family="serif", serif=["Computer Modern Roman"], size=20)
matplotlib.rc("text", usetex=True)


def make_plot(timestamps, values, labels, num):
    normals = labels == 0
    anomalies = labels == 1
    
    new_labels = np.zeros(labels.shape)
    new_labels[normals] = np.min(values)
    new_labels[anomalies] = np.max(values) - (np.max(values) - np.min(values)) * 0.9
    
    line_plot([timestamps, timestamps],
              [values, new_labels],
              fig_size=(8, 8),
              title=f"UCR series {num}")


reader = UCRReader("../data/anomaly_detection/ucr/")

ds = reader[12]
timestamp = ds["timestamp"].to_numpy()
values = ds["value"].to_numpy()
labels = ds["class"].to_numpy()
make_plot(timestamp, values, labels, 13)
make_plot(timestamp[15500:16500], values[15500:16500], labels[15500:16500], 13)

ds = reader[18]
timestamp = ds["timestamp"].to_numpy()
values = ds["value"].to_numpy()
labels = ds["class"].to_numpy()
make_plot(timestamp, values, labels, 19)
make_plot(timestamp[5700:6700], values[5700:6700], labels[5700:6700], 19)
