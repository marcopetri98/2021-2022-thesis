import itertools

import pandas as pd

from anomalearn.utils import print_header, print_step

# which and at which significance level to do the normality tests
EXPERIMENT = "voting"
F1_SCORING = False
if F1_SCORING:
    ADD = "f1_41_"
else:
    ADD = ""

# models involved in testing and path where scores are saved
MODELS = ["iforest"]
SAVE_DIR = "output/experiments_{}/confidence".format(EXPERIMENT)

for model_name in MODELS:
    print_header("Reducing confidence intervals CSV file for {}".format(model_name))

    SAVE_RESULTS_PATH = "{}/{}_{}_{}evaluation_bounds_reduced.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD)

    # load the experiments
    bounds_df = pd.read_csv("{}/{}_{}_{}evaluation_bounds.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD))
    datasets = bounds_df["dataset"].unique()
    train_lengths = bounds_df["training_length"].unique()
    methods = bounds_df[EXPERIMENT].unique()
    bounds_df.set_index(["dataset", "training_length", EXPERIMENT, "repetition"], inplace=True)
    
    if EXPERIMENT == "scoring":
        methods_cols = ["left", "centre", "right", "min", "max", "average", "non_overlapping"]
    else:
        methods_cols = ["left", "centre", "right", "majority_voting", "byzantine_voting", "unanimity", "voting"]

    # build results dataframe
    values = [["fridge1", "fridge2", "fridge3"],
              ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"],
              methods_cols]
    df_index = pd.MultiIndex.from_product(values, names=["dataset", "training_length", EXPERIMENT])
    results_df = pd.DataFrame(0.0, df_index, ["confidence_level",
                                              "lower_bound_val", "upper_bound_val", "sample_mean_val",
                                              "lower_bound_test", "upper_bound_test", "sample_mean_test"])

    for dataset, train_length, method in itertools.product(datasets, train_lengths, methods):
        print_step(f"Reducing confidence intervals on {dataset}, {train_length}, {method}")
        config_df = bounds_df.loc[dataset, train_length, method, 1]

        results_df.loc[(dataset, train_length, method), "confidence_level"] = config_df["confidence_level"]
        results_df.loc[(dataset, train_length, method), "lower_bound_val"] = config_df["lower_bound_val"]
        results_df.loc[(dataset, train_length, method), "upper_bound_val"] = config_df["upper_bound_val"]
        results_df.loc[(dataset, train_length, method), "sample_mean_val"] = config_df["sample_mean_val"]
        results_df.loc[(dataset, train_length, method), "lower_bound_test"] = config_df["lower_bound_test"]
        results_df.loc[(dataset, train_length, method), "upper_bound_test"] = config_df["upper_bound_test"]
        results_df.loc[(dataset, train_length, method), "sample_mean_test"] = config_df["sample_mean_test"]

    results_df.to_csv(SAVE_RESULTS_PATH)

    print_header(f"Ended reducing confidence intervals for {model_name}")
