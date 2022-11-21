from matplotlib import pyplot as plt, gridspec

from reader.time_series import MGABReader
from visualizer import line_plot

reader = MGABReader("data/anomaly_detection/mgab/")

for SERIES in range(10):
    ds = reader.read(SERIES).get_dataframe()

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    gs = gridspec.GridSpec(2, 1)

    series = fig.add_subplot(gs[0, 0])
    line_plot(ds["timestamp"].values,
              ds["value"].values,
              ax=series)

    targets = fig.add_subplot(gs[1, 0])
    line_plot(ds["timestamp"].values,
              ds["class"].values,
              ax=targets)

    plt.show()
