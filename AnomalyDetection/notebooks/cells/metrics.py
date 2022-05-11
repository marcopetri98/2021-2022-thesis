import json
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def compute_metrics(true_labels: np.ndarray,
                    scores: np.ndarray,
                    pred_labels: np.ndarray = None,
                    compute_roc_auc: bool = True,
                    only_roc_auc: bool = False):
    if not only_roc_auc:
        precision = metrics.precision_score(true_labels, pred_labels)
        recall = metrics.recall_score(true_labels, pred_labels)
        f1_score = metrics.f1_score(true_labels, pred_labels)
        accuracy = metrics.accuracy_score(true_labels, pred_labels)
        avg_precision = metrics.average_precision_score(true_labels, scores)
        pre, rec, _ = precision_recall_curve(true_labels, scores, pos_label=1)
        precision_recall_auc = metrics.auc(rec, pre)
        
        print("ACCURACY SCORE: ", accuracy)
        print("PRECISION SCORE: ", precision)
        print("RECALL SCORE: ", recall)
        print("F1 SCORE: ", f1_score)
        print("AVERAGE PRECISION SCORE: ", avg_precision)
        print("PRECISION-RECALL AUC SCORE: ", precision_recall_auc)
    
    if compute_roc_auc:
        roc_auc = roc_auc_score(true_labels, scores)
        print("AUROC SCORE: ", roc_auc)

def plot_roc_curve(true_labels: np.ndarray,
                   true_scores: np.ndarray,
                   pos_label: int = 1,
                   sample_weights: np.ndarray = None,
                   drop_intermediate: bool = True,
                   fig_size: Tuple = (6, 6),
                   curve_color: str = 'b') -> None:
    fpr, tpr, thresholds = roc_curve(true_labels,
                                     true_scores,
                                     pos_label=pos_label,
                                     sample_weight=sample_weights,
                                     drop_intermediate=drop_intermediate)
    curve_fmt = curve_color + '-'

    fig = plt.figure(figsize=fig_size)
    plt.plot([0, 1], [0, 1], 'k-', linewidth=0.5)
    plt.plot(fpr, tpr, curve_fmt, linewidth=0.5)
    plt.title("ROC Curve")
    plt.xlabel("Fallout [FP / (FP + TN)]")
    plt.ylabel("Recall [TP / (FN + TP)]")
    plt.show()

def plot_precision_recall_curve(true_labels: np.ndarray,
                                scores: np.ndarray,
                                pos_label: int = 1,
                                sample_weights: np.ndarray = None,
                                fig_size: Tuple = (6, 6),
                                curve_color: str = 'b') -> None:
    pre, rec, thresholds = precision_recall_curve(true_labels,
                                                  scores,
                                                  pos_label=pos_label,
                                                  sample_weight=sample_weights)
    curve_fmt = curve_color + '-'

    fig = plt.figure(figsize=fig_size)
    plt.plot([0, 1], [1, 0], 'k-', linewidth=0.5)
    plt.plot(rec, pre, curve_fmt, linewidth=0.5)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall [TP / (FN + TP)]")
    plt.ylabel("Precision [TP / (FP + TP)]")
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          fig_size: Tuple = (6, 6)) -> None:
    fig = plt.figure(figsize=fig_size)
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True, fmt=".0f")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted values")
    plt.ylabel("True values")
    plt.show()

def plot_univariate_time_series_predictions(dataframe: pd.DataFrame,
                                            predictions: np.ndarray,
                                            index_column: str = "timestamp",
                                            value_column: str = "value",
                                            target_column: str = "target",
                                            num_ticks: int = 5,
                                            fig_size: Tuple = (16, 15),
                                            data_color: str = 'k',
                                            label_color: str = 'r',
                                            pred_color: str = 'g') -> None:
    """Plot data with its labels and the relative predictions"""
    fictitious_idx, indexes, ticks = __compute_idx_ticks(dataframe,
                                                         num_ticks,
                                                         index_column)
    data_fmt, label_fmt, pred_fmt = __compute_formats(data_color,
                                                      label_color,
                                                      pred_color)

    fig, axs = plt.subplots(3, 1, figsize=fig_size)
    __plot_time_series_and_labels(dataframe,
                                  data_fmt,
                                  label_fmt,
                                  fictitious_idx,
                                  indexes,
                                  ticks,
                                  value_column,
                                  target_column,
                                  axs)

    axs[2].plot(fictitious_idx,
                predictions,
                pred_fmt,
                linewidth=0.5)
    axs[2].set_xticks(indexes, ticks)
    axs[2].set_title("Time series predictions")
    plt.show()

def plot_time_series_forecast(real_array: np.ndarray,
                              pred_array: np.ndarray,
                              num_ticks: int = 5,
                              fig_size: Tuple = (16, 6),
                              real_color: str = 'k',
                              pred_color: str = 'g',
                              on_same_plot: bool = True) -> None:
    fictitious_idx = np.arange(real_array.shape[0])
    indexes = np.linspace(0, real_array.shape[0] - 1, num_ticks, dtype=np.intc)
    ticks = indexes
    real_data_fmt = real_color + '-'
    pred_data_fmt = pred_color + '-'
    
    if on_same_plot:
        fig = plt.figure(figsize=fig_size)
        plt.plot(fictitious_idx,
                 real_array,
                 real_data_fmt,
                 linewidth=1)
        plt.plot(fictitious_idx,
                 pred_array,
                 pred_data_fmt,
                 linewidth=1)
        plt.xticks(indexes, ticks)
        plt.title("Time series data")
        plt.show()
    else:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        axs[0].plot(fictitious_idx,
                 real_array,
                 real_data_fmt,
                 linewidth=1)
        axs[0].set_xticks(indexes, ticks)
        axs[0].set_title("Time series data")
        
        axs[1].plot(fictitious_idx,
                 pred_array,
                 pred_data_fmt,
                 linewidth=1)
        axs[1].set_xticks(indexes, ticks)
        axs[1].set_title("Time series predictions")
        plt.show()

def plot_time_series_with_predicitons_bars(dataframe: pd.DataFrame,
                                           predictions: np.ndarray,
                                           bars: np.ndarray,
                                           index_column: str = "timestamp",
                                           value_column: str = "value",
                                           num_ticks: int = 5,
                                           fig_size: Tuple = (16, 5),
                                           data_color: str = 'k',
                                           pred_color: str = 'g') -> None:
    fictitious_idx, indexes, ticks = __compute_idx_ticks(dataframe,
                                                         num_ticks,
                                                         index_column)
    data_fmt, _, pred_fmt = __compute_formats(data_color,
                                              pred_color=pred_color,
                                              pred_type="point")
    anomalies = np.argwhere(predictions == 1)

    fig = plt.figure(figsize=fig_size)
    plt.plot(fictitious_idx,
             dataframe[value_column],
             data_fmt,
             linewidth=0.5)
    plt.scatter(fictitious_idx[anomalies],
                np.array(dataframe[value_column])[anomalies],
                c=pred_fmt)
    plt.vlines(fictitious_idx[bars],
               np.min(np.array(dataframe[value_column])),
               np.max(np.array(dataframe[value_column])),
               color=['r']*(fictitious_idx[bars]).size)
    plt.xticks(indexes, ticks)
    plt.title("Time series with predictions as dots")
    plt.show()

def get_windows_indices(dataframe: pd.DataFrame,
                        data_key: str,
                        combined_windows_path: str,
                        index_column: str = "timestamp") -> np.ndarray:
    file = open(combined_windows_path)
    combined_windows = json.load(file)
    file.close()

    desired_windows = combined_windows[data_key]
    desired_windows = [elem for window in desired_windows for elem in window]
    desired_windows = [elem[:-7] for elem in desired_windows]
    timestamps = dataframe[index_column]
    mask = np.zeros(np.array(dataframe[index_column]).size)
    for i in range(np.array(timestamps).size):
        if timestamps[i] in desired_windows:
            mask[i] = 1
    bool_mask = [True if val == 1 else False for val in mask]

    return np.argwhere(bool_mask)


def get_bars_indices_on_test_df(complete_df: pd.DataFrame,
                                dataframe: pd.DataFrame,
                                data_key: str,
                                combined_windows_path: str,
                                index_column: str = "timestamp") -> np.array:
    bars = get_windows_indices(complete_df, data_key, combined_windows_path, index_column)
    all_timestamps = complete_df["timestamp"].tolist()
    bars = [dataframe["timestamp"].tolist().index(all_timestamps[int(bar)])
            for bar in bars
            if all_timestamps[int(bar)] in dataframe["timestamp"].tolist()]
    return np.array(bars)

def __compute_idx_ticks(dataframe: pd.DataFrame,
                        num_ticks: int,
                        index_column: str):
    fictitious_idx = np.arange(dataframe.shape[0])
    indexes = np.linspace(0, dataframe.shape[0] - 1, num_ticks, dtype=np.intc)
    ticks = np.array(dataframe[index_column])[indexes] if index_column is not None else indexes
    return fictitious_idx, indexes, ticks


def __compute_formats(data_color: str,
                      label_color: str = None,
                      pred_color: str = 'g',
                      data_type: str = "line",
                      label_type: str = "line",
                      pred_type: str = "line") -> Tuple:
    d_sep = '-' if data_type == "line" else ''
    l_sep = '-' if label_type == "line" else ''
    p_sep = '-' if pred_type == "line" else ''
    data_fmt = data_color + d_sep
    label_fmt = label_color + l_sep if label_color is not None else None
    pred_fmt = pred_color + p_sep
    return data_fmt, label_fmt, pred_fmt


def __plot_time_series_and_labels(dataframe: pd.DataFrame,
                                  data_fmt: str,
                                  label_fmt: str,
                                  fictitious_idx: np.ndarray,
                                  indexes: np.ndarray,
                                  ticks: list,
                                  value_column: str,
                                  target_column: str,
                                  axs) -> None:
    axs[0].plot(fictitious_idx,
                dataframe[value_column],
                data_fmt,
                linewidth=0.5)
    axs[0].set_xticks(indexes, ticks)
    axs[0].set_title("Time series data")

    axs[1].plot(fictitious_idx,
                dataframe[target_column],
                label_fmt,
                linewidth=0.5)
    axs[1].set_xticks(indexes, ticks)
    axs[1].set_title("Time series labels")

def make_metric_plots(dataframe: pd.DataFrame,
                      true_labels: np.ndarray,
                      scores: np.ndarray,
                      pred_labels: np.ndarray):
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    plot_roc_curve(true_labels, scores)
    plot_precision_recall_curve(true_labels, scores)
    plot_confusion_matrix(confusion_matrix)
    plot_univariate_time_series_predictions(dataframe, pred_labels)
    plot_univariate_time_series_predictions(dataframe, scores)