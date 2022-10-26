import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from mleasy.applications import Zangrando2022Threshold, Zangrando2022Loader
from mleasy.utils import print_header, print_step


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
THRESHOLDS = 41
REPETITIONS = 10

for model_name in MODELS:
    print_header("Computing best F1 of {}".format(model_name))

    SAVE_RESULTS_PATH = "{}/{}_scoring_f1_{}_evaluation.csv".format(SAVE_DIR, model_name.lower(), THRESHOLDS)

    values = [["fridge1", "fridge2", "fridge3"],
              ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"],
              ["left", "centre", "right", "min", "max", "average", "non_overlapping"],
              [e for e in range(1, REPETITIONS+1)]]
    df_index = pd.MultiIndex.from_product(values, names=["dataset", "training_length", "scoring", "repetition"])
    results_df = pd.DataFrame(0.0, df_index, ["f1_val", "f1_test", "threshold"])

    data_loader = Zangrando2022Loader(DATASETS, TRAIN_LENGTH)

    for dataset in DATASETS:
        for train_length in TRAIN_LENGTH:
            for window_size in WINDOW_SIZE:
                for score_method in SCORINGS:
                    for rep in range(REPETITIONS):
                        print_step("Get data values and labels")
                        train_seq, train_all, valid, test = data_loader.get_train_valid_test(dataset,
                                                                                             train_length,
                                                                                             window_size,
                                                                                             window_multiple_of_period=False)

                        valid_scores_path = "{}/{}/validation_{}_{}_{}_{}_{}_rep{}_scores.csv".format(SAVE_DIR,
                                                                                                      model_name,
                                                                                                      model_name,
                                                                                                      dataset,
                                                                                                      score_method,
                                                                                                      train_length,
                                                                                                      window_size,
                                                                                                      rep + 1)
                        test_scores_path = "{}/{}/testing_{}_{}_{}_{}_{}_rep{}_scores.csv".format(SAVE_DIR,
                                                                                                  model_name,
                                                                                                  model_name,
                                                                                                  dataset,
                                                                                                  score_method,
                                                                                                  train_length,
                                                                                                  window_size,
                                                                                                  rep + 1)

                        print_step("The model is {} and the method is {}".format(model_name, score_method))
                        print_step("Reading validation and testing scores of repetition {}".format(rep + 1))
                        valid_df = pd.read_csv(valid_scores_path)
                        test_df = pd.read_csv(test_scores_path)

                        valid_scores = valid_df["anomaly_score"].values
                        test_scores = test_df["anomaly_score"].values

                        print_step("Computing the optimal threshold as in Zangrando2022")
                        if model_name == "lof":
                            threshold_comp = Zangrando2022Threshold(False, num_thresholds=THRESHOLDS)
                        else:
                            threshold_comp = Zangrando2022Threshold(True, num_thresholds=THRESHOLDS)

                        true_labels = valid["target"].values
                        true_labels, valid_scores = cut_true_pred_labels(true_labels, valid_scores, score_method, window_size)
                        best_f1, best_threshold = threshold_comp.compute_best_threshold(valid_scores, true_labels)

                        print_step("The best F1 on validation is {} and the best threshold is {}".format(best_f1, best_threshold))
                        results_df.loc[(dataset, train_length, score_method, rep + 1), "f1_val"] = best_f1
                        results_df.loc[(dataset, train_length, score_method, rep + 1), "threshold"] = best_threshold

                        print_step("Computing the F1 on the test")
                        true_labels = test["target"].values
                        true_labels, test_scores = cut_true_pred_labels(true_labels, test_scores, score_method, window_size)

                        anomalies = np.where(test_scores > best_threshold)
                        test_labels = np.zeros(test_scores.shape[0])
                        test_labels[anomalies] = 1
                        f1 = f1_score(true_labels, test_labels)

                        print_step("The best F1 on testing is {}".format(f1))
                        results_df.loc[(dataset, train_length, score_method, rep + 1), "f1_test"] = f1

    results_df.to_csv(SAVE_RESULTS_PATH)
    print_header("Ended F1 computation of {}".format(model_name))
