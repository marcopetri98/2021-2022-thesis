import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec

from visualizer import line_plot


def add_all_line_plots(fig_to_draw,
                       type_of_plot,
                       gs,
                       thresholds: list | int = 41,
                       voting_exp: str = "1_over_wind"):
    ax = None
    if type_of_plot == "scoring":
        metric_name = "AUROC"
        method_col = "scoring"
        metric_col = "auroc_test"
    elif type_of_plot == "voting":
        metric_name = "F1"
        method_col = "voting"
        metric_col = "f1_test"
    elif type_of_plot == "scoring_f1":
        metric_name = "F1"
        method_col = "scoring"
        metric_col = "f1_test"
    elif type_of_plot == "compare_scoring_f1":
        metric_name = "F1"
        method_col = "scoring"
        metric_col = "f1_test"
    else:
        metric_name = "F1"
        method_col = "scoring"
        metric_col = "f1_val"

    for i, model in enumerate(MODELS):
        if type_of_plot == "scoring":
            df = pd.read_csv("output/experiments_scoring/{}_scoring_evaluation.csv".format(model))
        elif type_of_plot == "voting":
            df = pd.read_csv("output/experiments_voting/{}_{}_voting_evaluation.csv".format(voting_exp, model))
        elif type_of_plot == "compare_scoring_f1":
            df = pd.read_csv("output/experiments_scoring/{}_scoring_f1_{}_evaluation.csv".format(model, thresholds[0]))
            df2 = pd.read_csv("output/experiments_scoring/{}_scoring_f1_{}_evaluation.csv".format(model, thresholds[1]))
        else:
            df = pd.read_csv("output/experiments_scoring/{}_scoring_f1_{}_evaluation.csv".format(model, thresholds))

        datasets = np.unique(df["dataset"])
        methods = np.unique(df[method_col])

        for j, dataset in enumerate(datasets):
            x_values = []
            y_values = []
            y_values2 = []
            x_ticks = []
            colors = []
            labels = []

            for x, score_method in enumerate(methods):
                plot_rows = df[df[method_col] == score_method]
                plot_rows = plot_rows[plot_rows["dataset"] == dataset]
                
                if type_of_plot == "compare_scoring_f1":
                    plot_rows2 = df2[df[method_col] == score_method]
                    plot_rows2 = plot_rows2[plot_rows2["dataset"] == dataset]
                    y_values2.append(plot_rows2[metric_col].values)

                x_ticks = plot_rows["training_length"].values.tolist()
                x_values.append(np.arange(10))
                y_values.append(plot_rows[metric_col].values)
                colors.append(COLORS[x])
                labels.append(score_method)

            ax = fig_to_draw.add_subplot(gs[i, j])
            line_plot(x_values,
                      y_values if type_of_plot != "compare_scoring_f1" else [e[0] - e[1] for e in zip(y_values, y_values2)],
                      x_ticks_loc=np.arange(10),
                      x_ticks_labels=x_ticks,
                      colors=colors,
                      series_labels=labels,
                      y_axis_label=metric_name,
                      x_axis_label="Training length",
                      title="{} {} on {}".format(model, metric_name, dataset),
                      ax=ax,
                      plot_legend=False)

    return ax


def make_fig_plot(type_of_plot,
                  legend_loc,
                  super_title,
                  thresholds: list | int = 41,
                  voting_exp: str = "1_over_wind"):
    # produce the line plot for scoring experiments
    fig = plt.figure(figsize=(15, 15), tight_layout=True)
    gs = gridspec.GridSpec(3, 3)

    ax = add_all_line_plots(fig, type_of_plot, gs, thresholds, voting_exp)

    handles, line_labels = ax.get_legend_handles_labels()
    fig.legend(handles, line_labels, loc=legend_loc)
    fig.suptitle(super_title)
    plt.show()


MODELS = ["iforest", "lof", "ocsvm"]
COLORS = ["red", "black", "blue", "green", "darkorange", "mediumvioletred", "darkturquoise"]

# make_fig_plot("scoring",
#               "lower right",
#               "Experiments on scoring methods for anomaly detection in time series using fridge datasets")
#
# make_fig_plot("voting",
#               "lower right",
#               "Experiments on voting methods for anomaly detection in time series using fridge datasets",
#               voting_exp="001")

# thresholds_ = 11
# make_fig_plot("scoring_f1",
#               "lower right",
#               f"Experiments on scoring (best test F1) methods trying {thresholds_} thresholds on validation",
#               thresholds_)
#
# make_fig_plot("scoring_f1_val",
#               "lower right",
#               f"Experiments on scoring (best validation F1) methods trying {thresholds_} thresholds on validation",
#               thresholds_)

thresholds_ = [11, 41]
make_fig_plot("compare_scoring_f1",
              "lower right",
              f"Test scoring difference between score selection of {thresholds_[0]} values against {thresholds_[1]}",
              thresholds_)
