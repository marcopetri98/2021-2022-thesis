import os.path

import numpy as np
import pandas as pd

from mleasy.visualizer import line_plot

this_folder = os.path.dirname(__file__)
fridge1_dataset_path = "./../data/anomaly_detection/private_fridge/fridge1/all_fridge1.csv"
fridge1_dataset_path = os.path.join(this_folder, fridge1_dataset_path)

fridge1_dataset = pd.read_csv(fridge1_dataset_path)

values_col = fridge1_dataset["device_consumption"].values
timestamps = fridge1_dataset["ctime"].values

START_IDX = 0
POINTS = 15000

print_values = values_col[START_IDX:START_IDX+POINTS]
times_values = timestamps[START_IDX:START_IDX+POINTS]

x_axis = np.arange(POINTS)
y_axis = print_values

ticks_loc = np.linspace(0, POINTS - 1, 5, dtype=np.intc)
ticks_lab = times_values[ticks_loc]

line_plot(x_axis,
          y_axis,
          x_ticks_loc=ticks_loc,
          x_ticks_labels=ticks_lab,
          x_ticks_rotation=15,
          title="Line plot on FRIDGE 1",
          x_axis_label="Timestamp",
          y_axis_label="Power consumption",
          fig_size=(18, 8))
