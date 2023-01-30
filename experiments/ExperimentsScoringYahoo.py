import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from anomalearn.applications import Munir2018Loader
from anomalearn.algorithms.models.time_series.anomaly.machine_learning import TSAIsolationForest, TSALOF, TSAOCSVM
from anomalearn.utils import print_header, print_step, print_warning


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


MODELS = ["iforest", "lof", "ocsvm"]
BENCHMARKS = ["a1", "a2", "a3", "a4"]
WINDOW_SIZE = [21]
SCORINGS = ["left", "centre", "right", "min", "max", "average", "non_overlapping"]
BENCHMARKS_SERIES = {"a1": range(67), "a2": range(100), "a3": range(100), "a4": range(100)}
TRAIN_LENGTH = [0.5, 0.33, 0.22, 0.11, 0.095, 0.079, 0.063, 0.048, 0.032, 0.016]
SAVE_DIR = "../output/experiments_scoring/yahoo"

for model_name in MODELS:
    print_header("Doing scoring experiments with {}".format(model_name))
    
    SAVE_RESULTS_PATH = "{}/{}_scoring_evaluation.csv".format(SAVE_DIR, model_name.lower())
    
    values = [["a1", "a2", "a3", "a4"],
              ["0.50", "0.33", "0.22", "0.11", "0.095", "0.079", "0.063", "0.048", "0.032", "0.016"],
              ["left", "centre", "right", "min", "max", "average", "non_overlapping"]]
    df_index = pd.MultiIndex.from_product(values, names=["benchmark", "training_length", "scoring"])
    results_df = pd.DataFrame(0.0, df_index, ["avg_auroc_val", "val_series", "avg_auroc_test", "test_series"])
    
    for benchmark in BENCHMARKS:
        for train_length in TRAIN_LENGTH:
            for window_size in WINDOW_SIZE:
                data_loader = Munir2018Loader(window_size,
                                              train_perc=train_length,
                                              valid_perc=0.05,
                                              test_perc=1 - train_length - 0.05)

                for score_method in SCORINGS:
                    total_val_auroc = 0
                    total_tst_auroc = 0
                    val_sum_of_auroc = 0
                    tst_sum_of_auroc = 0

                    for series in BENCHMARKS_SERIES[benchmark]:
                        train_seq, valid, test = data_loader.get_train_valid_test(series,
                                                                                  "yahoo_s5",
                                                                                  window_size)

                        train_pts = [e["value"].values.reshape(-1, 1) for e in train_seq]
                        train_lab = [e["target"].values for e in train_seq]

                        print_step(f"Training the model on series {series} of benchmark {benchmark}")
                        print_step(f"The length of the training set is {train_length}%")

                        if model_name == "iforest":
                            model = TSAIsolationForest(scoring=score_method if score_method != "non_overlapping" else "average",
                                                       classification="points_score",
                                                       random_state=22,
                                                       window=window_size,
                                                       n_estimators=70,
                                                       max_samples=400)
                            model.fit_multiple(train_pts, train_lab)
                        elif model_name == "lof":
                            model = TSALOF(scoring=score_method if score_method != "non_overlapping" else "average",
                                           classification="points_score",
                                           scaling="none",
                                           novelty=True,
                                           window=window_size,
                                           n_neighbors=50)
                            model.fit_multiple(train_pts, train_lab)
                        elif model_name == "ocsvm":
                            model = TSAOCSVM(scoring=score_method if score_method != "non_overlapping" else "average",
                                             classification="points_score",
                                             window=window_size,
                                             gamma=0.05,
                                             nu=0.001,
                                             tol=1e-8)
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

                        output_path = "{}/{}/validation_{}_{}_{}_{}_{}_{}_scores.csv".format(SAVE_DIR, model_name, model_name, benchmark, series, score_method, train_length, window_size)
                        scores_frame: pd.DataFrame = new_frame.copy()
                        scores_frame.insert(len(scores_frame.columns), "anomaly_score", scores)
                        scores_frame.to_csv(output_path, index=False)

                        print_step("Saved validation score to {}".format(output_path))
                        print_step("Computing ROC AUC for the validation")

                        true_labels, scores = cut_true_pred_labels(true_labels, scores, score_method, window_size)
                        if np.sum(true_labels) != 0:
                            roc_auc = roc_auc_score(true_labels, scores)
                            val_sum_of_auroc += roc_auc
                            total_val_auroc += 1

                            print_step("ROC AUC on validation is {}".format(roc_auc))
                        else:
                            print_warning("The validation set has no anomalies. AUROC cannot be computed")

                        print_step("Scoring the testing set")

                        data_test = test["value"].values.reshape(-1, 1)
                        true_labels = test["target"].values
                        scores = model.anomaly_score(data_test)

                        new_frame: pd.DataFrame = test.copy()
                        new_frame = new_frame.drop(columns=["target", "value"])

                        output_path = "{}/{}/testing_{}_{}_{}_{}_{}_{}_scores.csv".format(SAVE_DIR, model_name, model_name, benchmark, series, score_method, train_length, window_size)
                        scores_frame: pd.DataFrame = new_frame.copy()
                        scores_frame.insert(len(scores_frame.columns), "anomaly_score", scores)
                        scores_frame.to_csv(output_path, index=False)

                        print_step("Saved testing score to {}".format(output_path))
                        print_step("Computing ROC AUC for the testing")

                        true_labels, scores = cut_true_pred_labels(true_labels, scores, score_method, window_size)
                        if np.sum(true_labels) != 0:
                            roc_auc = roc_auc_score(true_labels, scores)
                            tst_sum_of_auroc += roc_auc
                            total_tst_auroc += 1

                            print_step("ROC AUC on testing is {}".format(roc_auc))
                        else:
                            print_warning("The testing set has no anomalies. AUROC cannot be computed")

                    avg_val_auroc = val_sum_of_auroc / total_val_auroc if total_val_auroc != 0 else "NaN"
                    avg_tst_auroc = tst_sum_of_auroc / total_tst_auroc if total_tst_auroc != 0 else "NaN"

                    results_df.loc[(benchmark, str(train_length), score_method), "avg_auroc_val"] = avg_val_auroc
                    results_df.loc[(benchmark, str(train_length), score_method), "avg_auroc_test"] = avg_tst_auroc
                    results_df.loc[(benchmark, str(train_length), score_method), "val_series"] = total_val_auroc
                    results_df.loc[(benchmark, str(train_length), score_method), "test_series"] = total_tst_auroc

    results_df.to_csv(SAVE_RESULTS_PATH)
    print_header("Ended experiments on {}".format(model_name))
