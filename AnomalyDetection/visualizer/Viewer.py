from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


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
	plt.plot([0,1], [0,1], 'k-', linewidth=0.5)
	plt.plot(fpr, tpr, curve_fmt, linewidth=0.5)
	plt.title("ROC Curve")
	plt.xlabel("Fallout [FP / (FP + TN)]")
	plt.ylabel("Recall [TP / (FN + TP)]")
	fig.show()

def plot_precision_recall_curve(true_labels: np.ndarray,
								true_scores: np.ndarray,
								pos_label: int = 1,
								sample_weights: np.ndarray = None,
								fig_size: Tuple = (6, 6),
								curve_color: str = 'b') -> None:
	pre, rec, thresholds = precision_recall_curve(true_labels,
												  true_scores,
												  pos_label=pos_label,
												  sample_weight=sample_weights)
	curve_fmt = curve_color + '-'
	
	fig = plt.figure(figsize=fig_size)
	plt.plot([0, 1], [1, 0], 'k-', linewidth=0.5)
	plt.plot(rec, pre, curve_fmt, linewidth=0.5)
	plt.title("Precision-Recall Curve")
	plt.xlabel("Recall [TP / (FN + TP)]")
	plt.ylabel("Precision [TP / (FP + TP)]")
	fig.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray,
						  fig_size: Tuple = (6, 6)) -> None:
	"""Plot in a single image the confusion matrix and the metrics"""
	fig = plt.figure(figsize=fig_size)
	sn.heatmap(confusion_matrix, cmap="Blues", annot=True, fmt=".0f")
	plt.title("Confusion matrix")
	plt.xlabel("Predicted values")
	plt.ylabel("True values")
	fig.show()

def plot_time_series_ndarray(array: np.ndarray,
							 num_ticks: int = 5,
							 fig_size: Tuple = (16, 6),
							 color: str = 'k') -> None:
	"""Plot the time series expressed as ndarray"""
	fictitious_idx = np.arange(array.shape[0])
	indexes = np.linspace(0, array.shape[0] - 1, num_ticks, dtype=np.intc)
	ticks = indexes
	data_fmt = color + '-'

	fig = plt.figure(figsize=fig_size)
	plt.plot(fictitious_idx,
			 array,
			 data_fmt,
			 linewidth=0.5)
	plt.xticks(indexes, ticks)
	plt.title("Time series data")
	fig.show()

def plot_univariate_time_series(dataframe: pd.DataFrame,
								index_column: str = "timestamp",
								value_column: str = "value",
								target_column: str = "target",
								num_ticks: int = 5,
								fig_size: Tuple = (16,12),
								data_color: str = 'k',
								label_color: str = 'r') -> None:
	"""Plot data and its labels."""
	fictitious_idx, indexes, ticks = __compute_idx_ticks(dataframe,
														 num_ticks,
														 index_column)
	data_fmt, label_fmt, _ = __compute_formats(data_color,
													  label_color)

	fig, axs = plt.subplots(2, 1, figsize=fig_size)
	__plot_time_series_and_labels(dataframe,
								  data_fmt,
								  label_fmt,
								  fictitious_idx,
								  indexes,
								  ticks,
								  value_column,
								  target_column,
								  axs)
	fig.show()

def plot_univariate_time_series_predictions(dataframe: pd.DataFrame,
											predictions: np.ndarray,
											index_column: str = "timestamp",
											value_column: str = "value",
											target_column: str = "target",
											num_ticks: int = 5,
											fig_size: Tuple = (16,15),
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
	fig.show()

def __compute_idx_ticks(dataframe: pd.DataFrame,
						num_ticks: int,
						index_column: str):
	fictitious_idx = np.arange(dataframe.shape[0])
	indexes = np.linspace(0, dataframe.shape[0] - 1, num_ticks, dtype=np.intc)
	ticks = dataframe[index_column][indexes] if index_column is not None else indexes
	return fictitious_idx, indexes, ticks

def __compute_formats(data_color: str,
					  label_color: str,
					  pred_color: str = 'g') -> Tuple:
	data_fmt = data_color + '-'
	label_fmt = label_color + '-'
	pred_fmt = pred_color + '-'
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