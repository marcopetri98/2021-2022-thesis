# External
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Project
import visualizer.Viewer as vw


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

def make_metric_plots(dataframe: pd.DataFrame,
					  true_labels: np.ndarray,
					  scores: np.ndarray,
					  pred_labels: np.ndarray):
	confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
	vw.plot_roc_curve(true_labels, scores)
	vw.plot_precision_recall_curve(true_labels, scores)
	vw.plot_confusion_matrix(confusion_matrix)
	vw.plot_univariate_time_series_predictions(dataframe, pred_labels)
	vw.plot_univariate_time_series_predictions(dataframe, scores)
