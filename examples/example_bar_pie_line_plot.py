import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from anomalearn.visualizer import line_plot, bar_class_distribution, pie_class_distribution

this_folder = os.path.dirname(__file__)

iris_dataset_path = "./../data/classification/iris/iris.csv"
iris_dataset_path = os.path.join(this_folder, iris_dataset_path)

iris_dataset = pd.read_csv(iris_dataset_path)

classes_col = iris_dataset["class"].values
classes = np.unique(classes_col)

num_classes = np.count_nonzero(classes)

fridge1_dataset_path = "./../data/anomaly_detection/private_fridge/fridge1/all_fridge1.csv"
fridge1_dataset_path = os.path.join(this_folder, fridge1_dataset_path)

fridge1_dataset = pd.read_csv(fridge1_dataset_path)

values_col = fridge1_dataset["device_consumption"].values
timestamps = fridge1_dataset["ctime"].values

print_values = values_col[:1000]
times_values = timestamps[:1000]

x_axis = np.arange(1000)
y_axis = print_values

ticks_loc = np.linspace(0, 999, 5, dtype=np.intc)
ticks_lab = times_values[ticks_loc]

classes_frequencies = np.zeros(num_classes)
for i, class_name in enumerate(classes):
    classes_frequencies[i] = np.count_nonzero(classes_col == class_name)
    
plt.rcParams.update({"font.size": 16, "font.family": "serif"})
fig = plt.figure(figsize=(16, 16), tight_layout=True)
gs = gridspec.GridSpec(2, 4)

ax_bar = fig.add_subplot(gs[0, 0:2])

bar_class_distribution(classes_frequencies,
                       classes.tolist(),
                       bars_colors=["orchid", "lightblue", "lightgreen"],
                       y_axis_label="Percentage",
                       x_axis_label="Class",
                       title="Iris data set class distribution",
                       ax=ax_bar)

ax_pie = fig.add_subplot(gs[0, 2:4])

pie_class_distribution(classes_frequencies,
                       classes.tolist(),
                       colors=["orchid", "lightblue", "lightgreen"],
                       title="Iris data set class distribution",
                       percentage_fmt="%.2f %%",
                       ax=ax_pie)

ax_line = fig.add_subplot(gs[1, 1:3])

line_plot(x_axis,
          y_axis,
          x_ticks_loc=ticks_loc,
          x_ticks_labels=ticks_lab,
          x_ticks_rotation=15,
          title="Line plot on FRIDGE 1",
          x_axis_label="Timestamp",
          y_axis_label="Power consumption",
          ax=ax_line)

plt.show()
