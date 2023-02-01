import numpy as np
from sklearn.metrics import confusion_matrix

from ..input_validation import check_array_1d


def _get_binary_confusion_elems(y_true,
                                y_pred) -> np.ndarray:
    """Gets true positives, false negatives, false positives and true negatives.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tn : float
        The number of true negatives.
        
    fp : float
        The number of false positives.
        
    fn : float
        The number of false negatives.
        
    tp : float
        The number of true positives.
    """
    check_array_1d(y_true, "y_true")
    check_array_1d(y_pred, "y_pred")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.ravel(confusion_matrix(y_true, y_pred))


def true_positive_rate(y_true,
                       y_pred) -> float:
    """Computes the True Positive Rate.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tpr : float
        The True Positive Rate.
    """
    tn, fp, fn, tp = _get_binary_confusion_elems(y_true, y_pred)
    return tp / (tp + fn)


def true_negative_rate(y_true,
                       y_pred) -> float:
    """Computes the True Negative Rate.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tnr : float
        The True Negative Rate.
    """
    tn, fp, fn, tp = _get_binary_confusion_elems(y_true, y_pred)
    return tn / (tn + fp)
