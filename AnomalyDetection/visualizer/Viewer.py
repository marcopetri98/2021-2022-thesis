from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_univariate_time_series(dataframe: pd.DataFrame,
								index_column: str = "timestamp",
								value_column: str = "value",
								target_column: str = "target",
								num_ticks: int = 5,
								fig_size: Tuple = (16,12),
								data_color: str = 'k',
								label_color: str = 'r') -> None:
	"""Plot data and its labels."""
	fictitious_idx = np.arange(dataframe.shape[0])
	indexes = np.linspace(0, dataframe.shape[0] - 1, num_ticks, dtype=np.intc)
	ticks = dataframe[index_column][indexes] if index_column is not None else indexes
	data_fmt = data_color + '-'
	label_fmt = label_color + '-'

	fig, axs = plt.subplots(2, 1, figsize=fig_size)
	axs[0].set_title("Time series data")
	axs[0].plot(fictitious_idx,
				dataframe[value_column],
				data_fmt,
				linewidth=0.5)
	axs[0].set_xticks(indexes, ticks)
	
	axs[1].set_title("Time series labels")
	axs[1].plot(fictitious_idx,
				dataframe[target_column],
				label_fmt,
				linewidth=0.5)
	axs[1].set_xticks(indexes, ticks)
	plt.show()

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
	fictitious_idx = np.arange(dataframe.shape[0])
	indexes = np.linspace(0, dataframe.shape[0] - 1, num_ticks, dtype=np.intc)
	ticks = dataframe[index_column][indexes] if index_column is not None else indexes
	data_fmt = data_color + '-'
	label_fmt = label_color + '-'
	pred_fmt = pred_color + '-'
	
	fig, axs = plt.subplots(3, 1, figsize=fig_size)
	axs[0].set_title("Time series data")
	axs[0].plot(fictitious_idx,
				dataframe[value_column],
				data_fmt,
				linewidth=0.5)
	axs[0].set_xticks(indexes, ticks)
	
	axs[1].set_title("Time series labels")
	axs[1].plot(fictitious_idx,
				dataframe[target_column],
				label_fmt,
				linewidth=0.5)
	axs[1].set_xticks(indexes, ticks)
	
	axs[2].set_title("Time series predictions")
	axs[2].plot(fictitious_idx,
				predictions,
				pred_fmt,
				linewidth=0.5)
	axs[2].set_xticks(indexes, ticks)
	plt.show()
