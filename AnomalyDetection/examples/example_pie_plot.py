import os.path

import numpy as np
import pandas as pd

from mleasy.visualizer import pie_class_distribution

this_folder = os.path.dirname(__file__)
iris_dataset_path = "./../data/classification/iris/iris.csv"
iris_dataset_path = os.path.join(this_folder, iris_dataset_path)

iris_dataset = pd.read_csv(iris_dataset_path)

classes_col = iris_dataset["class"].values
classes = np.unique(classes_col)

num_classes = np.count_nonzero(classes)

classes_frequencies = np.zeros(num_classes)
for i, class_name in enumerate(classes):
    classes_frequencies[i] = np.count_nonzero(classes_col == class_name)

pie_class_distribution(classes_frequencies,
                       classes.tolist(),
                       colors=["orchid", "lightblue", "lightgreen"],
                       title="Iris data set class distribution",
                       percentage_fmt="%.2f %%")
