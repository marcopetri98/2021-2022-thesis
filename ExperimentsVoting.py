import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from mleasy.applications import Zangrando2022Loader
from mleasy.models.time_series.anomaly.machine_learning import TSAIsolationForest, TSALOF, TSAOCSVM
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

    return true, pred

MODELS = ["iforest", "lof", "ocsvm"]
DATASETS = ["fridge1", "fridge2", "fridge3"]
WINDOW_SIZE = [21]
TRAIN_LENGTH = ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"]
VOTINGS = ["left", "centre", "right", "majority_voting", "byzantine_voting", "unanimity", "voting"]
VOTING_STEP = 0.01

for model_name in MODELS:
    print_header("Doing scoring experiments with {}".format(model_name))
    
    SAVE_RESULTS_PATH = "output/experiments_voting/{}_voting_evaluation.csv".format(model_name.lower())
    
    values = [["fridge1", "fridge2", "fridge3"],
              ["1m", "3w", "2w", "1w", "6d", "5d", "4d", "3d", "2d", "1d"],
              ["left", "centre", "right", "majority_voting", "byzantine_voting", "unanimity", "voting"]]
    df_index = pd.MultiIndex.from_product(values, names=["dataset", "training_length", "voting"])
    results_df = pd.DataFrame(0.0, df_index, ["f1_val", "f1_test", "threshold"])

    data_loader = Zangrando2022Loader(DATASETS, TRAIN_LENGTH)
    
    for dataset in DATASETS:
        for train_length in TRAIN_LENGTH:
            for window_size in WINDOW_SIZE:
                for vote_method in VOTINGS:
                    train_seq, train_all, valid, test = data_loader.get_train_valid_test(dataset,
                                                                                         train_length,
                                                                                         window_size,
                                                                                         window_multiple_of_period=False)
    
                    train_pts = [e["value"].values.reshape(-1, 1) for e in train_seq]
                    train_lab = [e["target"].values for e in train_seq]
    
                    print_step("Training the model")

                    if vote_method == "byzantine_voting":
                        voting_params = {"classification": "voting", "threshold": 2/3}
                    else:
                        voting_params = {"classification": vote_method}

                    if model_name == "iforest":
                        model = TSAIsolationForest(random_state=22,
                                                   window=window_size,
                                                   n_estimators=70,
                                                   max_samples=400,
                                                   **voting_params)
                        model.fit_multiple(train_pts, train_lab)
                    elif model_name == "lof":
                        model = TSALOF(scaling="none",
                                       novelty=True,
                                       window=window_size,
                                       n_neighbors=50,
                                       **voting_params)
                        model.fit_multiple(train_pts, train_lab)
                    elif model_name == "ocsvm":
                        model = TSAOCSVM(window=window_size,
                                         gamma=0.05,
                                         nu=0.001,
                                         tol=1e-8,
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

                    output_path = "output/experiments_voting/{}/validation_{}_{}_{}_{}_{}_labels.csv".format(model_name, model_name, dataset, vote_method, train_length, window_size)
                    labels_frame: pd.DataFrame = new_frame.copy()
                    labels_frame.insert(len(labels_frame.columns), "anomaly_label", labels)
                    labels_frame.to_csv(output_path, index=False)
                    
                    print_step("Saved validation labels to {}".format(output_path))
                    print_step("Computing F1 for the validation")

                    true_labels, labels = cut_true_pred_labels(true_labels, labels, vote_method, window_size)
                    f1 = f1_score(true_labels, labels)
                    results_df.loc[(dataset, train_length, vote_method), "f1_val"] = f1
                    if vote_method in ["voting", "byzantine_voting", "unanimity", "majority_voting"]:
                        results_df.loc[(dataset, train_length, vote_method), "threshold"] = model.threshold
                    else:
                        results_df.loc[(dataset, train_length, vote_method), "threshold"] = "NaN"

                    print_step("F1 on validation is {}".format(f1))
                    print_step("Saved validation F1")
                    print_step("Classifying the testing set")

                    data_test = test["value"].values.reshape(-1, 1)
                    true_labels = test["target"].values
                    labels = model.classify(data_test)

                    new_frame: pd.DataFrame = test.copy()
                    new_frame = new_frame.drop(columns=["target", "value"])

                    output_path = "output/experiments_voting/{}/testing_{}_{}_{}_{}_{}_labels.csv".format(model_name, model_name, dataset, vote_method, train_length, window_size)
                    scores_frame: pd.DataFrame = new_frame.copy()
                    scores_frame.insert(len(scores_frame.columns), "anomaly_label", labels)
                    scores_frame.to_csv(output_path, index=False)
                    
                    print_step("Saved testing labels to {}".format(output_path))
                    print_step("Computing F1 for the testing")
                    
                    true_labels, labels = cut_true_pred_labels(true_labels, labels, vote_method, window_size)
                    f1 = f1_score(true_labels, labels)
                    results_df.loc[(dataset, train_length, vote_method), "f1_test"] = f1

                    print_step("F1 on testing is {}".format(f1))
                
    results_df.to_csv(SAVE_RESULTS_PATH)
    print_header("Ended experiments on {}".format(model_name))
