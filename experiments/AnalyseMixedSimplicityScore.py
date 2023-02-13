import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from anomalearn.analysis import analyse_mixed_simplicity
from anomalearn.reader.time_series import YahooS5Reader, NASAReader, NABReader, UCRReader, MGABReader, SMDReader, \
    KitsuneReader, GHLReader, ExathlonReader, rts_config


def get_series_columns(series):
    is_multivariate = rts_config["Univariate"]["value_column"] not in series.columns
    if is_multivariate:
        series_cols = [e for e in series.columns if rts_config["Multivariate"]["channel_column"] in e]
    else:
        series_cols = [rts_config["Univariate"]["value_column"]]
    return is_multivariate, series_cols


if __name__ == "__main__":
    this_folder = Path(__file__).parent
    data_folder = this_folder / "../data/anomaly_detection"
    # create all the readers
    yahoo_reader = YahooS5Reader(benchmark_location=f"{str(data_folder)}/yahoo_s5/")
    ucr_reader = UCRReader(benchmark_location=f"{str(data_folder)}/ucr/")
    smd_reader = SMDReader(benchmark_location=f"{str(data_folder)}/smd/")
    nab_reader = NABReader(benchmark_location=f"{str(data_folder)}/nab/")
    mgab_reader = MGABReader(benchmark_location=f"{str(data_folder)}/mgab/")
    kitsune_reader = KitsuneReader(benchmark_location=f"{str(data_folder)}/kitsune/")
    ghl_reader = GHLReader(benchmark_location=f"{str(data_folder)}/ghl/")
    exathlon_reader = ExathlonReader(benchmark_location=f"{str(data_folder)}/exathlon/")

    nasa_msl_smap = NASAReader(anomalies_path=f"{str(data_folder)}/nasa_msl_smap/labeled_anomalies.csv")

    # create lists of readers to iterate over them
    light_readers = [yahoo_reader, ucr_reader, smd_reader, nab_reader, mgab_reader,
                     ghl_reader, nasa_msl_smap, exathlon_reader]
    light_names = ["Yahoo", "UCR", "SMD", "NAB", "MGAB", "GHL", "NASA", "Exathlon"]
    heavy_readers = [kitsune_reader]
    heavy_names = ["Kitsune"]

    # create the dataframe to store the results
    output_folder = this_folder / "../output/datasets_simplicity"
    output_file = output_folder / "simplicity_scores.csv"
    all_readers = light_readers + heavy_readers
    all_datasets = light_names + heavy_names
    
    try:
        results_df = pd.read_csv(output_file, index_col="series")
    except FileNotFoundError:
        results_df = pd.DataFrame(None,
                                  pd.Index(np.arange(sum([len(r) for r in all_readers])),
                                           name="series"),
                                  ["Dataset",
                                   "Mixed score",
                                   "Constant score",
                                   "Moving average score",
                                   "Moving standard deviation score",
                                   "Analysis result dict"],
                                  dtype=object)

    for idx, reader in enumerate(all_readers):
        print(f"Reading time series using {all_datasets[idx]}...", end="\n\n")
        
        for ser_idx in range(len(reader)):
            row_idx = ser_idx + sum([len(all_readers[i]) for i in range(idx)]) if idx > 0 else ser_idx

            if isinstance(results_df.loc[row_idx, "Dataset"], float) and math.isnan(results_df.loc[row_idx, "Dataset"]):
                series = reader.read(ser_idx, verbose=False).get_dataframe()
                is_multivariate, series_cols = get_series_columns(series)
                
                print(f"Reading series number {ser_idx + 1}...")
                print("Getting the labels and the time series from dataframe...")
                
                if all_datasets[idx] == "GHL":
                    time_series = series[series.columns.intersection(series_cols)].values
                    l1 = np.asarray(series["class_0"].values, dtype=np.int32)
                    l2 = np.asarray(series["class_1"].values, dtype=np.int32)
                    l3 = np.asarray(series["class_2"].values, dtype=np.int32)
                    time_series_labels = l1 | l2 | l3
                else:
                    time_series = series[series.columns.intersection(series_cols)].values
                    time_series_labels = series[rts_config["Univariate"]["target_column"]].values
                
                if time_series.ndim == 1:
                    time_series = time_series.reshape((-1, 1))
                
                time_series = np.ascontiguousarray(time_series, dtype=np.double)
                time_series_labels = np.ascontiguousarray(time_series_labels, dtype=np.int32)
                
                if len(np.unique(time_series_labels)) == 2:
                    print("Analysing the mixed simplicity score...")
                    start_time = time.time()
                    mixed_analysis = analyse_mixed_simplicity(time_series, time_series_labels)
                    end_time = time.time()
                    print(f"Time series analysed in {end_time - start_time:.2f}s")
                    print(f"The dataset MIXED SCORE IS: {mixed_analysis['mixed_score']}")
                    print(f"Saving current results to file...", end="\n\n")
                    
                    results_df.loc[row_idx, "Dataset"] = all_datasets[idx]
                    results_df.loc[row_idx, "Mixed score"] = mixed_analysis["mixed_score"]
                    results_df.loc[row_idx, "Constant score"] = mixed_analysis["const_result"]["constant_score"]
                    results_df.loc[row_idx, "Moving average score"] = mixed_analysis["mov_avg_result"]["mov_avg_score"]
                    results_df.loc[row_idx, "Moving standard deviation score"] = mixed_analysis["mov_std_result"]["mov_std_score"]
                    results_df.loc[row_idx, "Analysis result dict"] = str(mixed_analysis)
                else:
                    print("The time series has only one label...")
                    print("Nothing can be computed with such a configuration", end="\n\n")
                    
                    results_df.loc[row_idx, "Dataset"] = all_datasets[idx]
                    results_df.loc[row_idx, "Mixed score"] = None
                    results_df.loc[row_idx, "Constant score"] = None
                    results_df.loc[row_idx, "Moving average score"] = None
                    results_df.loc[row_idx, "Moving standard deviation score"] = None
                    results_df.loc[row_idx, "Analysis result dict"] = None
                
                results_df.to_csv(output_file)
