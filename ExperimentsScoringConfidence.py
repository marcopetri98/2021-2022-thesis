import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mleasy.applications import Zangrando2022Loader
from mleasy.algorithms.models.time_series.anomaly.machine_learning import TSAIsolationForest
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
    elif cutting == "non_overlapping":
        num_of_nan = np.sum(np.isnan(pred))
        true = true[:-num_of_nan]
        pred = pred[:-num_of_nan]

    return true, pred


MODELS = ["iforest"]
DATASETS = ["fridge1", "fridge2", "fridge3"]
WINDOW_SIZE = [21]
TRAIN_LENGTH = ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"]
SCORINGS = ["left", "centre", "right", "min", "max", "average", "non_overlapping"]
SAVE_DIR = "output/experiments_scoring/confidence"
# never put a value less than 1
BASE_TO_ADD = 1
REPETITIONS = 25
EXPERIMENT_REP = 1
SEEDS: list | None = load_py_json("ExperimentsConfidenceSeeds.json")

if len(SEEDS) < BASE_TO_ADD + REPETITIONS - 1:
    raise ValueError("There aren't enough seeds. Increase them.")

for model_name in MODELS:
    print_header("Doing scoring experiments with {}".format(model_name))
    
    SAVE_RESULTS_PATH = "{}/{}_scoring_evaluation_rep{}.csv".format(SAVE_DIR, model_name.lower(), EXPERIMENT_REP)
    
    values = [["fridge1", "fridge2", "fridge3"],
              ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"],
              ["left", "centre", "right", "min", "max", "average", "non_overlapping"],
              [e for e in range(1, REPETITIONS+BASE_TO_ADD)]]
    df_index = pd.MultiIndex.from_product(values, names=["dataset", "training_length", "scoring", "repetition"])
    results_df = pd.DataFrame(0.0, df_index, ["seed", "auroc_val", "auroc_test"])

    data_loader = Zangrando2022Loader(DATASETS, TRAIN_LENGTH)
    
    for dataset, train_length, window_size, score_method, rep in itertools.product(DATASETS, TRAIN_LENGTH, WINDOW_SIZE, SCORINGS, range(REPETITIONS)):
        train_seq, train_all, valid, test = data_loader.get_train_valid_test(dataset,
                                                                             train_length,
                                                                             window_size,
                                                                             window_multiple_of_period=False)

        train_pts = [e["value"].values.reshape(-1, 1) for e in train_seq]
        train_lab = [e["target"].values for e in train_seq]

        print_step("Performing the repetition number {} for this configuration".format(rep + BASE_TO_ADD))
        print_step("Training the model")

        if model_name == "iforest":
            model = TSAIsolationForest(scoring=score_method if score_method != "non_overlapping" else "average",
                                       classification="points_score",
                                       window=window_size,
                                       n_estimators=70,
                                       max_samples=400,
                                       random_state=SEEDS[rep + BASE_TO_ADD - 1])
            model.fit_multiple(train_pts, train_lab)
        else:
            raise ValueError("Model {} is not expected".format(model_name))

        data_test = valid["value"].values.reshape(-1, 1)
        true_labels = valid["target"].values

        print_step("The scoring method is {}".format(score_method))
        print_step("Scoring the validation set")

        if score_method == "non_overlapping":
            model.set_params(stride=window_size)

        scores = model.anomaly_score(data_test)

        new_frame: pd.DataFrame = valid.copy()
        new_frame = new_frame.drop(columns=["target", "value"])

        output_path = "{}/{}/validation_{}_{}_{}_{}_{}_rep{}_scores.csv".format(SAVE_DIR,
                                                                                model_name,
                                                                                model_name,
                                                                                dataset,
                                                                                score_method,
                                                                                train_length,
                                                                                window_size,
                                                                                rep + BASE_TO_ADD)
        scores_frame: pd.DataFrame = new_frame.copy()
        scores_frame.insert(len(scores_frame.columns), "anomaly_score", scores)
        scores_frame.to_csv(output_path, index=False)

        print_step("Saved validation score to {}".format(output_path))
        print_step("Computing ROC AUC for the validation")

        true_labels, scores = cut_true_pred_labels(true_labels, scores, score_method, window_size)
        roc_auc = roc_auc_score(true_labels, scores)
        results_df.loc[(dataset, train_length, score_method, rep + BASE_TO_ADD), "auroc_val"] = roc_auc

        print_step("ROC AUC on validation is {}".format(roc_auc))
        print_step("Saved validation ROC AUC")
        print_step("Scoring the testing set")

        data_test = test["value"].values.reshape(-1, 1)
        true_labels = test["target"].values
        scores = model.anomaly_score(data_test)

        new_frame: pd.DataFrame = test.copy()
        new_frame = new_frame.drop(columns=["target", "value"])

        output_path = "{}/{}/testing_{}_{}_{}_{}_{}_rep{}_scores.csv".format(SAVE_DIR,
                                                                             model_name,
                                                                             model_name,
                                                                             dataset,
                                                                             score_method,
                                                                             train_length,
                                                                             window_size,
                                                                             rep + BASE_TO_ADD)
        scores_frame: pd.DataFrame = new_frame.copy()
        scores_frame.insert(len(scores_frame.columns), "anomaly_score", scores)
        scores_frame.to_csv(output_path, index=False)

        print_step("Saved testing score to {}".format(output_path))
        print_step("Computing ROC AUC for the testing")

        true_labels, scores = cut_true_pred_labels(true_labels, scores, score_method, window_size)
        roc_auc = roc_auc_score(true_labels, scores)
        results_df.loc[(dataset, train_length, score_method, rep + BASE_TO_ADD), "auroc_test"] = roc_auc

        # save the seed on the CSV
        results_df.loc[(dataset, train_length, score_method, rep + BASE_TO_ADD), "seed"] = SEEDS[rep + BASE_TO_ADD - 1]

        print_step("ROC AUC on testing is {}".format(roc_auc))

    results_df.to_csv(SAVE_RESULTS_PATH)
    print_header("Ended experiments on {}".format(model_name))
