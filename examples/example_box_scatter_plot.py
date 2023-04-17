import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

from anomalearn.visualizer import scatter_plot, box_plot

this_folder = os.path.dirname(__file__)
iris_dataset_path = "./../data/classification/iris/iris.csv"
iris_dataset_path = os.path.join(this_folder, iris_dataset_path)

iris_dataset = pd.read_csv(iris_dataset_path)

iris_dataset_setosa = iris_dataset.loc[iris_dataset["class"] == "Iris-setosa"]
iris_dataset_versicolor = iris_dataset.loc[iris_dataset["class"] == "Iris-versicolor"]
iris_dataset_virginica = iris_dataset.loc[iris_dataset["class"] == "Iris-virginica"]

sequences = []
sequences.append(iris_dataset_setosa["sepal_length_cm"].values.tolist())
sequences.append(iris_dataset_versicolor["sepal_length_cm"].values.tolist())
sequences.append(iris_dataset_virginica["sepal_length_cm"].values.tolist())

x_seq = []
y_seq = []
x_seq.append(iris_dataset_setosa["sepal_length_cm"].values.tolist())
x_seq.append(iris_dataset_versicolor["sepal_length_cm"].values.tolist())
x_seq.append(iris_dataset_virginica["sepal_length_cm"].values.tolist())
y_seq.append(iris_dataset_setosa["sepal_width_cm"].values.tolist())
y_seq.append(iris_dataset_versicolor["sepal_width_cm"].values.tolist())
y_seq.append(iris_dataset_virginica["sepal_width_cm"].values.tolist())

plt.rcParams.update({"font.size": 16, "font.family": "serif"})
fig = plt.figure(figsize=(16, 8), tight_layout=True)
gs = gridspec.GridSpec(1, 2)

ax_box = fig.add_subplot(gs[0, 0])

box_plot(sequences,
         ["Setosa", "Versicolor", "Virginica"],
         title="Iris data set box plot",
         y_axis_label="Sepal length",
         colors=["orchid", "lightblue", "lightgreen"],
         ax=ax_box)

ax_sca = fig.add_subplot(gs[0, 1])

scatter_plot(x_seq,
             y_seq,
             colors=["orchid", "lightblue", "lightgreen"],
             markers=["o", "^", "x"],
             title="Iris data set scatter plot",
             y_axis_label="Sepal width",
             x_axis_label="Sepal length",
             labels=["Setosa", "Versicolor", "Virginica"],
             ax=ax_sca)

plt.show()
