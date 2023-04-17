import os

import pandas as pd

from anomalearn.visualizer import box_plot

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

box_plot(sequences,
         ["Setosa", "Versicolor", "Virginica"],
         title="Iris data set box plot",
         y_axis_label="Sepal length",
         colors=["orchid", "lightblue", "lightgreen"])
