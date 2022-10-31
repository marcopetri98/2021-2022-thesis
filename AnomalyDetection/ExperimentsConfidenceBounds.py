import itertools
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import t

from utils import print_header, print_step

# which and at which significance level to do the normality tests
EXPERIMENT = "scoring"
ALPHA = 0.05
F1_SCORING = False
if F1_SCORING:
    ADD = "f1_41_"
else:
    ADD = ""

# models involved in testing and path where scores are saved
MODELS = ["iforest"]
SAVE_DIR = "output/experiments_{}/confidence".format(EXPERIMENT)

for model_name in MODELS:
    print_header("Computing confidence intervals for {}".format(model_name))
    print_step("Computing the confidence intervals using the t-student for unknown mean and variance")

    SAVE_RESULTS_PATH = "{}/{}_{}_{}evaluation_bounds.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD)

    # load the experiments
    results_df = pd.read_csv("{}/{}_{}_{}evaluation_tested.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD))
    datasets = results_df["dataset"].unique()
    train_lengths = results_df["training_length"].unique()
    methods = results_df[EXPERIMENT].unique()
    results_df.set_index(["dataset", "training_length", EXPERIMENT, "repetition"], inplace=True)

    # copy results and add normality test column
    results_df.insert(len(results_df.columns), "confidence_level", "NaN")
    results_df.insert(len(results_df.columns), "lower_bound_val", "NaN")
    results_df.insert(len(results_df.columns), "upper_bound_val", "NaN")
    results_df.insert(len(results_df.columns), "sample_mean_val", "NaN")
    results_df.insert(len(results_df.columns), "lower_bound_test", "NaN")
    results_df.insert(len(results_df.columns), "upper_bound_test", "NaN")
    results_df.insert(len(results_df.columns), "sample_mean_test", "NaN")

    for dataset, train_length, method in itertools.product(datasets, train_lengths, methods):
        print_step(f"Computing confidence intervals on {dataset}, {train_length}, {method}")
        config_df = results_df.loc[dataset, train_length, method]

        if EXPERIMENT == "voting" or F1_SCORING:
            val_col = "f1_val"
            test_col = "f1_test"
        else:
            val_col = "auroc_val"
            test_col = "auroc_test"

        val_points = config_df[val_col].values
        val_is_normal = config_df["from_normal_val"]
        test_points = config_df[test_col].values
        test_is_normal = config_df["from_normal_test"]

        print_step("Computing the bounds and the estimate")

        val_sample_std = np.std(val_points, ddof=1)
        test_sample_std = np.std(test_points, ddof=1)
        val_sample_mean = np.mean(val_points)
        test_sample_mean = np.mean(test_points)

        print_step(f"Validation sample std is {val_sample_std} and sample mean is {val_sample_mean}")
        print_step(f"Validation sample std is {test_sample_std} and sample mean is {test_sample_mean}")

        # right-tail probability
        t_quantile = - t.ppf(ALPHA / 2, val_points.shape[0] - 1)

        print_step(f"The t-student quantile for {ALPHA / 2} with ddof {val_points.shape[0] - 1} is {t_quantile}")

        val_lower = val_sample_mean - t_quantile * (val_sample_std / sqrt(val_points.shape[0]))
        val_upper = val_sample_mean + t_quantile * (val_sample_std / sqrt(val_points.shape[0]))

        test_lower = test_sample_mean - t_quantile * (test_sample_std / sqrt(test_points.shape[0]))
        test_upper = test_sample_mean + t_quantile * (test_sample_std / sqrt(test_points.shape[0]))

        print_step(f"The confidence interval for validation is ({val_lower}, {val_upper})")
        print_step(f"The confidence interval for testing is ({test_lower}, {test_upper})")

        results_df.loc[(dataset, train_length, method), "confidence_level"] = 1 - ALPHA
        results_df.loc[(dataset, train_length, method), "lower_bound_val"] = val_lower
        results_df.loc[(dataset, train_length, method), "upper_bound_val"] = val_upper
        results_df.loc[(dataset, train_length, method), "sample_mean_val"] = val_sample_mean
        results_df.loc[(dataset, train_length, method), "lower_bound_test"] = test_lower
        results_df.loc[(dataset, train_length, method), "upper_bound_test"] = test_upper
        results_df.loc[(dataset, train_length, method), "sample_mean_test"] = test_sample_mean

    results_df.to_csv(SAVE_RESULTS_PATH)

    print_header(f"Ended computing confidence intervals for {model_name}")
