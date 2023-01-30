import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec

from anomalearn.visualizer import confidence_line_plot

# which and at which significance level to do the normality tests
EXPERIMENT = "scoring"
F1_SCORING = False
CORRECT_X_SCALE = True
SAME_Y_SCALE = True
Y_MIN = 0.66
Y_MAX = 0.98
if F1_SCORING:
    ADD = "f1_41_"
else:
    ADD = ""
    
METRIC = "AUROC" if EXPERIMENT == "scoring" and not F1_SCORING else "F1"

# constants for the plot
VALIDATION = False
LEGEND_LOC = "lower right"
SUPER_TITLE = f"{'Testing' if not VALIDATION else 'Validation'} confidence intervals for the {EXPERIMENT} methods"
COLORS = ["red", "black", "blue", "green", "darkorange", "mediumvioletred", "darkturquoise"]

# models involved in testing and path where scores are saved
MODELS = ["iforest"]
SAVE_DIR = "output/experiments_{}/confidence".format(EXPERIMENT)

for model_name in MODELS:
    if VALIDATION:
        lower_col = "lower_bound_val"
        upper_col = "upper_bound_val"
        mean_col = "sample_mean_val"
    else:
        lower_col = "lower_bound_test"
        upper_col = "upper_bound_test"
        mean_col = "sample_mean_test"

    # load the experiments
    bounds_df = pd.read_csv("{}/{}_{}_{}evaluation_bounds_reduced.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD))
    datasets = bounds_df["dataset"].unique()
    train_lengths = bounds_df["training_length"].unique()
    methods = bounds_df[EXPERIMENT].unique()

    SUPER_TITLE += f" at {bounds_df.iloc[0]['confidence_level'] * 100}% confidence level"

    # produce the line plot for scoring experiments
    fig = plt.figure(figsize=(15, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)
    
    for ds_idx, dataset in enumerate(datasets):
        x_values = []
        y_values = []
        x_ticks = []
        colors = []
        labels = []
        pos = np.array([0, 7, 14, 21, 22, 23, 24, 25, 26, 27]) if CORRECT_X_SCALE else np.arange(10)
        y_lim = {"low": Y_MIN, "high": Y_MAX} if SAME_Y_SCALE else None
        
        for met_idx, method in enumerate(methods):
            plot_df = bounds_df[bounds_df[EXPERIMENT] == method]
            plot_df = plot_df[plot_df["dataset"] == dataset]
            lowers = plot_df[lower_col].values
            uppers = plot_df[upper_col].values
            estimates = plot_df[mean_col].values

            x_ticks = plot_df["training_length"].values.tolist()
            x_values.append(pos)
            y_values.append([lowers, uppers, estimates])
            colors.append(COLORS[met_idx])
            labels.append(method)
    
        ax = fig.add_subplot(gs[0, ds_idx])
        confidence_line_plot(x_values,
                             y_values,
                             y_lim=y_lim,
                             x_ticks_loc=pos,
                             x_ticks_labels=x_ticks,
                             x_ticks_rotation=90 if CORRECT_X_SCALE else 0,
                             colors=colors,
                             series_labels=labels,
                             y_axis_label=f"Confidence interval for {METRIC}",
                             x_axis_label="Training length",
                             title="{} {} on {}".format(model_name, METRIC, dataset),
                             ax=ax,
                             plot_legend=False)

    handles, line_labels = ax.get_legend_handles_labels()
    fig.legend(handles, line_labels, loc=LEGEND_LOC)
    fig.suptitle(SUPER_TITLE)
    plt.show()
