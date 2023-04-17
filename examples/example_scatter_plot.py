import os

import pandas as pd

from anomalearn.visualizer import scatter_plot

this_folder = os.path.dirname(__file__)
iris_dataset_path = "./../data/classification/iris/iris.csv"
iris_dataset_path = os.path.join(this_folder, iris_dataset_path)

iris_dataset = pd.read_csv(iris_dataset_path)

iris_dataset_setosa = iris_dataset.loc[iris_dataset["class"] == "Iris-setosa"]
iris_dataset_versicolor = iris_dataset.loc[iris_dataset["class"] == "Iris-versicolor"]
iris_dataset_virginica = iris_dataset.loc[iris_dataset["class"] == "Iris-virginica"]

x_seq = []
y_seq = []
x_seq.append(iris_dataset_setosa["sepal_length_cm"].values.tolist())
x_seq.append(iris_dataset_versicolor["sepal_length_cm"].values.tolist())
x_seq.append(iris_dataset_virginica["sepal_length_cm"].values.tolist())
y_seq.append(iris_dataset_setosa["sepal_width_cm"].values.tolist())
y_seq.append(iris_dataset_versicolor["sepal_width_cm"].values.tolist())
y_seq.append(iris_dataset_virginica["sepal_width_cm"].values.tolist())

scatter_plot(x_seq,
             y_seq,
             colors=["orchid", "lightblue", "lightgreen"],
             markers=["o", "^", "x"],
             title="Iris data set scatter plot",
             y_axis_label="Sepal width",
             x_axis_label="Sepal length",
             labels=["Setosa", "Versicolor", "Virginica"])
