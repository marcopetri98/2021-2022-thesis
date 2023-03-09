import numpy as np

from anomalearn.utils import mov_avg, mov_std
from anomalearn.visualizer import line_plot


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

line_plot([np.arange(length)] * 4,
          [walk, mov_mean, mov_mean + mov_dev, mov_mean + mov_dev + constant],
          fig_size=(16, 8))
