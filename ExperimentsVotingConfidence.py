import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from mleasy.applications import Zangrando2022Loader
from mleasy.models.time_series.anomaly.machine_learning import TSAIsolationForest
from mleasy.utils import print_header, print_step, load_py_json


def cut_true_pred_labels(true, pred, cutting, window):
    # eliminate not scoreable points
    if cutting == "left":
        true = true[:-(window - 1)]
        pred = pred[:-(window - 1)]
    elif cutting == "right":
        true = true[window - 1:]
        pred = pred[window - 1:]
    elif cutting == "centre":
        true = true[int((window - 1) / 2):-int((window - 1) / 2)]
        pred = pred[int((window - 1) / 2):-int((window - 1) / 2)]

    return true, pred


MODELS = ["iforest"]
DATASETS = ["fridge1", "fridge2", "fridge3"]
WINDOW_SIZE = [21]
TRAIN_LENGTH = ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"]
VOTINGS = ["left", "centre", "right", "majority_voting", "byzantine_voting", "unanimity", "voting"]
VOTING_STEP = 0.01
SAVE_DIR = "output/experiments_voting/confidence"
# never put a value less than 1
BASE_TO_ADD = 1
REPETITIONS = 25
EXPERIMENT_REP = 1
SEEDS: list | None = load_py_json("ExperimentsConfidenceSeeds.json")

if len(SEEDS) < BASE_TO_ADD + REPETITIONS - 1:
    raise ValueError("There aren't enough seeds. Increase them.")

for model_name in MODELS:
    print_header("Doing scoring experiments with {}".format(model_name))
    
    SAVE_RESULTS_PATH = "{}/{}_voting_evaluation_rep{}.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT_REP)
    
    values = [["fridge1", "fridge2", "fridge3"],
              ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"],
              ["left", "centre", "right", "majority_voting", "byzantine_voting", "unanimity", "voting"],
              [e for e in range(1, REPETITIONS+BASE_TO_ADD)]]
    df_index = pd.MultiIndex.from_product(values, names=["dataset", "training_length", "voting", "repetition"])
    results_df = pd.DataFrame(0.0, df_index, ["seed", "f1_val", "f1_test", "threshold"])

    data_loader = Zangrando2022Loader(DATASETS, TRAIN_LENGTH)
    
    for dataset, train_length, window_size, vote_method, rep in itertools.product(DATASETS, TRAIN_LENGTH, WINDOW_SIZE, VOTINGS, range(REPETITIONS)):
        train_seq, train_all, valid, test = data_loader.get_train_valid_test(dataset,
                                                                             train_length,
                                                                             window_size,
                                                                             window_multiple_of_period=False)

        train_pts = [e["value"].values.reshape(-1, 1) for e in train_seq]
        train_lab = [e["target"].values for e in train_seq]

        print_step("Performing the repetition number {} for this configuration".format(rep + BASE_TO_ADD))
        print_step("Training the model")

        if vote_method == "byzantine_voting":
            voting_params = {"classification": "voting", "threshold": 2 / 3}
        else:
            voting_params = {"classification": vote_method}

        if model_name == "iforest":
            model = TSAIsolationForest(window=window_size,
                                       n_estimators=70,
                                       max_samples=400,
                                       random_state=SEEDS[rep + BASE_TO_ADD - 1],
                                       **voting_params)
            model.fit_multiple(train_pts, train_lab)
        else:
            raise ValueError("Model {} is not expected".format(model_name))

        print_step("The classification method is {}".format(vote_method))
        print_step("Classifying the validation set")

        data_test = valid["value"].values.reshape(-1, 1)
        true_labels = valid["target"].values
        labels = model.classify(data_test)

        # compute the best threshold for voting if voting has been chosen
        if vote_method == "voting":
            print_step("Computing the optimal voting threshold on validation")

            # compute the best voting threshold on the validation set
            best_f1 = -1
            best_threshold = -1
            for threshold in np.arange(0, 1 + VOTING_STEP, VOTING_STEP):
                model.threshold = threshold
                true_labels_copy = true_labels.copy()
                labels = model.classify(data_test)

                true_labels_copy, labels = cut_true_pred_labels(true_labels_copy, labels, vote_method, window_size)

                f1 = f1_score(true_labels_copy, labels)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            print_step("The best voting threshold is {}".format(best_threshold))
            model.threshold = best_threshold
            labels = model.classify(data_test)

        new_frame: pd.DataFrame = valid.copy()
        new_frame = new_frame.drop(columns=["target", "value"])

        output_path = "{}/{}/validation_{}_{}_{}_{}_{}_rep{}_labels.csv".format(SAVE_DIR,
                                                                                model_name,
                                                                                model_name,
                                                                                dataset,
                                                                                vote_method,
                                                                                train_length,
                                                                                window_size,
                                                                                rep + BASE_TO_ADD)
        labels_frame: pd.DataFrame = new_frame.copy()
        labels_frame.insert(len(labels_frame.columns), "anomaly_label", labels)
        labels_frame.to_csv(output_path, index=False)

        print_step("Saved validation labels to {}".format(output_path))
        print_step("Computing F1 for the validation")

        true_labels, labels = cut_true_pred_labels(true_labels, labels, vote_method, window_size)
        f1 = f1_score(true_labels, labels)
        results_df.loc[(dataset, train_length, vote_method, rep + BASE_TO_ADD), "f1_val"] = f1
        if vote_method in ["voting", "byzantine_voting", "unanimity", "majority_voting"]:
            results_df.loc[(dataset, train_length, vote_method, rep + BASE_TO_ADD), "threshold"] = model.threshold
        else:
            results_df.loc[(dataset, train_length, vote_method, rep + BASE_TO_ADD), "threshold"] = "NaN"

        print_step("F1 on validation is {}".format(f1))
        print_step("Saved validation F1")
        print_step("Classifying the testing set")

        data_test = test["value"].values.reshape(-1, 1)
        true_labels = test["target"].values
        labels = model.classify(data_test)

        new_frame: pd.DataFrame = test.copy()
        new_frame = new_frame.drop(columns=["target", "value"])

        output_path = "{}/{}/testing_{}_{}_{}_{}_{}_rep{}_labels.csv".format(SAVE_DIR,
                                                                             model_name,
                                                                             model_name,
                                                                             dataset,
                                                                             vote_method,
                                                                             train_length,
                                                                             window_size,
                                                                             rep + BASE_TO_ADD)
        scores_frame: pd.DataFrame = new_frame.copy()
        scores_frame.insert(len(scores_frame.columns), "anomaly_label", labels)
        scores_frame.to_csv(output_path, index=False)

        print_step("Saved testing labels to {}".format(output_path))
        print_step("Computing F1 for the testing")

        true_labels, labels = cut_true_pred_labels(true_labels, labels, vote_method, window_size)
        f1 = f1_score(true_labels, labels)
        results_df.loc[(dataset, train_length, vote_method, rep + BASE_TO_ADD), "f1_test"] = f1

        # save the seed on the CSV
        results_df.loc[(dataset, train_length, vote_method, rep + BASE_TO_ADD), "seed"] = SEEDS[rep + BASE_TO_ADD - 1]

        print_step("F1 on testing is {}".format(f1))
                
    results_df.to_csv(SAVE_RESULTS_PATH)
    print_header("Ended experiments on {}".format(model_name))
