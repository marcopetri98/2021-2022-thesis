import itertools

import pandas as pd
from scipy.stats import shapiro

from utils import print_header, print_step

# which and at which significance level to do the normality tests
EXPERIMENT = "voting"
ALPHA = 0.01
F1_SCORING = False
if F1_SCORING:
    ADD = "f1_41_"
else:
    ADD = ""

# models involved in testing and path where scores are saved
MODELS = ["iforest"]
SAVE_DIR = "output/experiments_{}/confidence".format(EXPERIMENT)

for model_name in MODELS:
    print_header("Doing normality tests on {}".format(model_name))

    SAVE_RESULTS_PATH = "{}/{}_{}_{}evaluation_tested.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD)

    # load the experiments
    results_df = pd.read_csv("{}/{}_{}_{}evaluation.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD))
    datasets = results_df["dataset"].unique()
    train_lengths = results_df["training_length"].unique()
    methods = results_df[EXPERIMENT].unique()
    results_df.set_index(["dataset", "training_length", EXPERIMENT, "repetition"], inplace=True)

    # copy results and add normality test column
    results_df.insert(len(results_df.columns), "statistic_val", "NaN")
    results_df.insert(len(results_df.columns), "pvalue_val", "NaN")
    results_df.insert(len(results_df.columns), "from_normal_val", "NaN")
    results_df.insert(len(results_df.columns), "statistic_test", "NaN")
    results_df.insert(len(results_df.columns), "pvalue_test", "NaN")
    results_df.insert(len(results_df.columns), "from_normal_test", "NaN")
    results_df.insert(len(results_df.columns), "or_normal", "NaN")

    for dataset, train_length, method in itertools.product(datasets, train_lengths, methods):
        print_step(f"Doing normality tests on {dataset}, {train_length}, {method}")
        config_df = results_df.loc[dataset, train_length, method]

        if EXPERIMENT == "voting" or F1_SCORING:
            val_col = "f1_val"
            test_col = "f1_test"
        else:
            val_col = "auroc_val"
            test_col = "auroc_test"

        val_points = config_df[val_col].values
        test_points = config_df[test_col].values

        print_step("Computing the Shapiro-Wilk test")
        val_statistic, val_pvalue = shapiro(val_points)
        test_statistic, test_pvalue = shapiro(test_points)

        print_step(f"Validation test statistic is {val_statistic} and p-value is {val_pvalue}")
        print_step(f"Testing test statistic is {test_statistic} and p-value is {test_pvalue}")

        # accept or refuse null hypothesis
        val_normal = "no" if val_pvalue < ALPHA else "yes"
        test_normal = "no" if test_pvalue < ALPHA else "yes"
        or_normal = "yes" if val_normal == "yes" or test_normal == "yes" else "no"

        # write the values on the dataframe
        results_df.loc[(dataset, train_length, method), "statistic_val"] = val_statistic
        results_df.loc[(dataset, train_length, method), "pvalue_val"] = val_pvalue
        results_df.loc[(dataset, train_length, method), "from_normal_val"] = val_normal
        results_df.loc[(dataset, train_length, method), "statistic_test"] = test_statistic
        results_df.loc[(dataset, train_length, method), "pvalue_test"] = test_pvalue
        results_df.loc[(dataset, train_length, method), "from_normal_test"] = test_normal
        results_df.loc[(dataset, train_length, method), "or_normal"] = or_normal

    results_df.to_csv(SAVE_RESULTS_PATH)

    print_header(f"Ended doing normality tests on {model_name}")
