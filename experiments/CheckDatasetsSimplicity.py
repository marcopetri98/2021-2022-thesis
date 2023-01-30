import itertools
import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from anomalearn.algorithms.models.time_series import TSAConstant, TSAMovAvgStd, TSAConstAvgStd
from anomalearn.reader.time_series import YahooS5Reader, NASAReader, NABReader, UCRReader, MGABReader, SMDReader, \
    KitsuneReader, GHLReader, ExathlonReader, rts_config

TEST_PERC = 0.2
UNUSABLE_TAG = "UNUSABLE"
MAX_WINDOW = 200


def get_series_columns(series):
    is_multivariate = rts_config["Univariate"]["value_column"] not in series.columns
    if is_multivariate:
        series_cols = [e for e in series.columns if rts_config["Multivariate"]["channel_column"] in e]
    else:
        series_cols = [rts_config["Univariate"]["value_column"]]
    return is_multivariate, series_cols


def get_train_test_models(series, series_cols):
    print(f"Train-test split will be {100 * (1 - TEST_PERC)}-{100 * TEST_PERC} "
          f"split will be used")

    train_end = round(series.shape[0] * (1 - TEST_PERC))
    train = series[0:train_end]
    test = series[train_end:]

    train_series = train[train.columns.intersection(series_cols)]
    train_labels = train[rts_config["Univariate"]["target_column"]]
    test_series = test[test.columns.intersection(series_cols)]
    test_labels = test[rts_config["Univariate"]["target_column"]]

    return train_series, train_labels, test_series, test_labels


def get_training_testing_diff_absdiff(version, train_series, train_labels, test_series, test_labels):
    match version:
        case 0:
            training = train_series.values
            testing = test_series.values
            training_labels = train_labels.values
            testing_labels = test_labels.values

        case 1:
            training = np.diff(train_series.values, axis=0)
            testing = np.diff(test_series.values, axis=0)
            training_labels = train_labels.iloc[1:].values
            testing_labels = test_labels.iloc[1:].values

        case _:
            training = np.abs(np.diff(train_series.values, axis=0))
            testing = np.abs(np.diff(test_series.values, axis=0))
            training_labels = train_labels.iloc[1:].values
            testing_labels = test_labels.iloc[1:].values
    return training, training_labels, testing, testing_labels


def classify_and_score(model, series, series_labels):
    pred = model.classify(series, verbose=2)
    half = int(np.count_nonzero(np.isnan(pred)) / 2)
    
    if half != 0:
        f1 = f1_score(series_labels[half:-half], pred[half:-half])
        pr = precision_score(series_labels[half:-half], pred[half:-half])
        rec = recall_score(series_labels[half:-half], pred[half:-half])
    else:
        f1 = f1_score(series_labels, pred)
        pr = precision_score(series_labels, pred)
        rec = recall_score(series_labels, pred)
        
    return pr, rec, f1


if __name__ == "__main__":
    data_folder = "data/anomaly_detection"
    # create all the readers
    yahoo_reader = YahooS5Reader(benchmark_location=f"{data_folder}/yahoo_s5/")
    ucr_reader = UCRReader(benchmark_location=f"{data_folder}/ucr/")
    smd_reader = SMDReader(benchmark_location=f"{data_folder}/smd/")
    nab_reader = NABReader(benchmark_location=f"{data_folder}/nab/")
    mgab_reader = MGABReader(benchmark_location=f"{data_folder}/mgab/")
    kitsune_reader = KitsuneReader(benchmark_location=f"{data_folder}/kitsune/")
    ghl_reader = GHLReader(benchmark_location=f"{data_folder}/ghl/")
    exathlon_reader = ExathlonReader(benchmark_location=f"{data_folder}/exathlon/")

    nasa_msl_smap = NASAReader(anomalies_path=f"{data_folder}/nasa_msl_smap/labeled_anomalies.csv")

    # create lists of readers to iterate over them
    light_readers = [yahoo_reader, ucr_reader, smd_reader, nab_reader, mgab_reader,
                     ghl_reader, exathlon_reader, nasa_msl_smap]
    light_names = ["Yahoo", "UCR", "SMD", "NAB", "MGAB", "GHL", "Exathlon", "NASA"]
    heavy_readers = [kitsune_reader]
    heavy_names = ["Kitsune"]

    # create the dataframe to store the results
    output_folder = "output/datasets_simplicity"
    output_file = f"{output_folder}/tested_models_and_datasets.csv"
    output_usable_series = f"{output_folder}/usable_series.csv"
    output_unusable_series = f"{output_folder}/unusable_series.csv"
    output_parameters = f"{output_folder}/learnt_parameters.csv"
    all_readers = light_readers + heavy_readers
    all_datasets = light_names + heavy_names
    settings = ["train", "test"]
    metrics = ["Pr", "Rec", "F1"]
    df_index = pd.MultiIndex.from_product([all_datasets, settings, metrics],
                                          names=["dataset", "setting", "metric"])
    models = ["Constant", "ConstantDiff", "ConstantAbsDiff",
              "MovingAvg", "MovingAvgDiff", "MovingAvgAbsDiff",
              "MovingStd", "MovingStdDiff", "MovingStdAbsDiff",
              "Keogh", "KeoghDiff", "KeoghAbsDiff"]
    results_df = pd.DataFrame(0.0, df_index, models)
    usable_df = results_df.copy()
    unusable_df = results_df.copy()
    try:
        parameters_df = pd.read_csv(output_parameters, index_col="series")
    except FileNotFoundError:
        parameters_index = pd.MultiIndex.from_product([all_datasets])
        parameters_df = pd.DataFrame(None,
                                     pd.Index(np.arange(sum([len(r) for r in all_readers])), name="series"),
                                     ["Dataset"] + models,
                                     dtype=object)

    # create list of models to try
    models_to_use = [TSAConstant(),
                     TSAMovAvgStd(max_window=MAX_WINDOW, method="movavg"),
                     TSAMovAvgStd(max_window=MAX_WINDOW, method="movstd"),
                     TSAConstAvgStd(max_window=MAX_WINDOW)]

    for idx, reader in enumerate(all_readers):
        print(f"Reading time series using {all_datasets[idx]}")

        for ser_idx, series in enumerate(reader):
            is_multivariate, series_cols = get_series_columns(series)
            train_series, train_labels, test_series, test_labels = get_train_test_models(series, series_cols)
            
            row_idx = ser_idx + sum([len(all_readers[i]) for i in range(idx)]) if idx > 0 else ser_idx
            
            for mod_idx, model in enumerate(models_to_use):
                for version in ["normal", "diff", "absdiff"]:
                    offset = 0 if version == "normal" else 1 if version == "diff" else 2
                    true_idx = mod_idx * 3 + offset
                    training, training_labels, testing, testing_labels = get_training_testing_diff_absdiff(offset,
                                                                                                           train_series,
                                                                                                           train_labels,
                                                                                                           test_series,
                                                                                                           test_labels)

                    try:
                        parameters = parameters_df.loc[row_idx, models[true_idx]]
                        if parameters != UNUSABLE_TAG and isinstance(parameters, float) and math.isnan(parameters):
                            # if the model is moving average or keogh adjust maximum window dimension
                            if mod_idx != 0:
                                max_window = min(round(testing.shape[0] / 2), MAX_WINDOW)
                                model.set_parameters(max_window=max_window)

                            model.fit(training, training_labels)
                        else:
                            if parameters == UNUSABLE_TAG:
                                raise ValueError("supervised training requires at least one anomaly")
                            
                            parameters = eval(parameters)
                            model.set_parameters(**parameters)

                        print("Classify the train and the test series to compute the f1")

                        train_pr, train_rec, train_f1 = classify_and_score(model, training, training_labels)
                        test_pr, test_rec, test_f1 = classify_and_score(model, testing, testing_labels)

                        results_df.loc[(all_datasets[idx], "train", "Pr")][models[true_idx]] += train_pr
                        results_df.loc[(all_datasets[idx], "train", "Rec")][models[true_idx]] += train_rec
                        results_df.loc[(all_datasets[idx], "train", "F1")][models[true_idx]] += train_f1
                        results_df.loc[(all_datasets[idx], "test", "Pr")][models[true_idx]] += test_pr
                        results_df.loc[(all_datasets[idx], "test", "Rec")][models[true_idx]] += test_rec
                        results_df.loc[(all_datasets[idx], "test", "F1")][models[true_idx]] += test_f1

                        usable_df.loc[(all_datasets[idx], "train", "Pr")][models[true_idx]] += 1
                        usable_df.loc[(all_datasets[idx], "train", "Rec")][models[true_idx]] += 1
                        usable_df.loc[(all_datasets[idx], "train", "F1")][models[true_idx]] += 1
                        usable_df.loc[(all_datasets[idx], "test", "Pr")][models[true_idx]] += 1
                        usable_df.loc[(all_datasets[idx], "test", "Rec")][models[true_idx]] += 1
                        usable_df.loc[(all_datasets[idx], "test", "F1")][models[true_idx]] += 1

                        parameters_df.loc[row_idx]["Dataset"] = all_datasets[idx]
                        parameters_df.loc[row_idx][models[true_idx]] = model.get_parameters()
                    except ValueError as e:
                        if str(e) != "supervised training requires at least one anomaly":
                            raise e
                        unusable_df.loc[(all_datasets[idx], "train", "Pr")][models[true_idx]] += 1
                        unusable_df.loc[(all_datasets[idx], "train", "Rec")][models[true_idx]] += 1
                        unusable_df.loc[(all_datasets[idx], "train", "F1")][models[true_idx]] += 1
                        unusable_df.loc[(all_datasets[idx], "test", "Pr")][models[true_idx]] += 1
                        unusable_df.loc[(all_datasets[idx], "test", "Rec")][models[true_idx]] += 1
                        unusable_df.loc[(all_datasets[idx], "test", "F1")][models[true_idx]] += 1

                        parameters_df.loc[row_idx]["Dataset"] = all_datasets[idx]
                        parameters_df.loc[row_idx][models[true_idx]] = UNUSABLE_TAG

                    results_df.to_csv(output_file)
                    usable_df.to_csv(output_usable_series)
                    unusable_df.to_csv(output_unusable_series)
                    parameters_df.to_csv(output_parameters)

    for dataset, setting, metric, model in itertools.product(all_datasets, settings, metrics, models):
        if usable_df.loc[(dataset, setting, metric), model] != 0:
            results_df.loc[(dataset, setting, metric), model] /= usable_df.loc[(dataset, setting, metric), model]
