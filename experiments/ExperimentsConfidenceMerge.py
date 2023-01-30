import itertools

import pandas as pd
from scipy.stats import shapiro

from utils import print_header, print_step

# which and at which significance level to do the normality tests
EXPERIMENT = "voting"
EXPERIMENT_REPETITIONS = 1
F1_SCORING = False
if F1_SCORING:
    ADD = "f1_41_"
else:
    ADD = ""

# models involved in testing and path where scores are saved
MODELS = ["iforest"]
SAVE_DIR = "output/experiments_{}/confidence".format(EXPERIMENT)

for model_name in MODELS:
    print_header("Mix multiple experiments together for {}".format(model_name))

    SAVE_RESULTS_PATH = "{}/{}_{}_{}evaluation.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD)

    # load the experiments
    experiments = [pd.read_csv("{}/{}_{}_{}evaluation_rep{}.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT, ADD, exp_rep))
                   for exp_rep in range(1, EXPERIMENT_REPETITIONS + 1)]
    datasets = experiments[0]["dataset"].unique()
    train_lengths = experiments[0]["training_length"].unique()
    methods = experiments[0][EXPERIMENT].unique()
    experiments = [e.set_index(["dataset", "training_length", EXPERIMENT, "repetition"]) for e in experiments]

    results_df = experiments[-1].copy()

    for dataset, train_length, method, exp_rep in itertools.product(datasets, train_lengths, methods, reversed(range(EXPERIMENT_REPETITIONS - 1))):
        print_step(f"Mixing experiments results for {dataset}, {train_length}, {method}")
        config_df = experiments[exp_rep].loc[dataset, train_length, method]
        exp_repetitions = config_df.index.values

        # update final result
        for col in results_df.columns:
            results_df.loc[(dataset, train_length, method, exp_repetitions), col] = config_df[col].values

    results_df.to_csv(SAVE_RESULTS_PATH)

    print_header(f"Ended mixing multiple experiments for {model_name}")
