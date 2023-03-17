import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from anomalearn.visualizer import scatter_plot, line_plot

matplotlib.rc("font", family="serif", serif=["Computer Modern Roman"], size=12)
matplotlib.rc("text", usetex=True)

rng = np.random.default_rng(seed=941)
normal_points = rng.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], size=50)
anomalous_1 = rng.multivariate_normal([4, 0], [[0.1, 0], [0, 0.1]], size=5)
anomalous_2 = rng.multivariate_normal([-4, 0], [[0.1, 0], [0, 0.1]], size=5)
anomalous_3 = rng.multivariate_normal([0, 4], [[0.1, 0], [0, 0.1]], size=5)
anomalous_4 = rng.multivariate_normal([0, -4], [[0.1, 0], [0, 0.1]], size=5)

# complete division
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot()
scatter_plot([normal_points[:, 0], anomalous_1[:, 0], anomalous_2[:, 0], anomalous_3[:, 0], anomalous_4[:, 0]],
             [normal_points[:, 1], anomalous_1[:, 1], anomalous_2[:, 1], anomalous_3[:, 1], anomalous_4[:, 1]],
             colors=["b", "r", "r", "r", "r"],
             ax=ax)
line_plot([[3.7, 3.7], [-3.6, -3.6], [-4.6, 4.7], [-4.6, 4.7]],
          [[-4.8, 5.3], [-4.8, 5.3], [-3.7, -3.7], [3.7, 3.7]],
          colors=["r"] * 4,
          ax=ax)
plt.title("Simple dataset with all boundaries")
plt.show()

# partial division
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot()
scatter_plot([normal_points[:, 0], anomalous_1[:, 0], anomalous_2[:, 0], anomalous_3[:, 0]],
             [normal_points[:, 1], anomalous_1[:, 1], anomalous_2[:, 1], anomalous_3[:, 1]],
             colors=["b", "r", "r", "r", "r"],
             ax=ax)
line_plot([[3.7, 3.7], [-3.6, -3.6], [-4.6, 4.7]],
          [[-4.8, 5.3], [-4.8, 5.3], [3.7, 3.7]],
          colors=["r"] * 3,
          ax=ax)
plt.title("Simple dataset with some boundaries")
plt.show()
