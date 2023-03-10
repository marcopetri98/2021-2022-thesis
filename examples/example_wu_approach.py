import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from anomalearn.utils import mov_avg, mov_std
from anomalearn.visualizer import line_plot


matplotlib.rc("font", family="serif", serif=["Computer Modern Roman"], size=12)
matplotlib.rc("text", usetex=True)


def random_walk(length: int, interval: tuple[int, int] = (-1, 1), seed: int | None = None):
    rng = np.random.default_rng(seed=seed)
    changes = rng.uniform(interval[0], interval[1], size=length)
    return np.cumsum(changes)


length = 1000
window = 50
constant = 2.5

walk = random_walk(length, (-2, 2), 152)
mov_mean = mov_avg(walk, window).reshape(-1)
mov_dev = mov_std(walk, window).reshape(-1)

# example figure of wu et al. approach
line_plot([np.arange(length)] * 4,
          [walk, mov_mean, mov_mean + mov_dev, mov_mean + mov_dev + constant],
          title="Example of statistical one-liner for triviality",
          fig_size=(10, 4))

# example of improvement of wu et al. approach
line_plot([np.arange(length)] * 6,
          [walk, mov_mean, mov_mean + mov_dev, mov_mean + mov_dev + constant, mov_mean - mov_dev, mov_mean - mov_dev - constant],
          title="Example of improvement with lower bound",
          fig_size=(10, 4))

walk_anti_diff = np.cumsum(walk)
walk_diff = np.diff(walk_anti_diff)

# example of series generating the random walk
line_plot(np.arange(length), walk_anti_diff, title="Original series", fig_size=(10, 4))

# example of first difference not stationary
line_plot(np.arange(length), np.append(walk_diff[0], walk_diff), title="First difference of the series", fig_size=(10, 4))
