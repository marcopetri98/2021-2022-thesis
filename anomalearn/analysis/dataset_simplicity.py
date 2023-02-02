import math
from typing import Callable, Any, Optional

import numpy as np
from sklearn.utils import check_array, check_X_y

from ..exceptions import ClosedOpenRangeError, SelectionError
from ..utils import true_positive_rate, true_negative_rate, mov_avg, mov_std


def _find_constant_score(x: np.ndarray,
                         y: np.ndarray) -> dict:
    """Find the constant score on this dataset.
    
    This method is exhaustive and find the exact constant score. If the constant
    score is 1 the dataset is constant simple.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        The time series to analyse.
    
    y : ndarray of shape (n_samples,)
        The labels of the time series.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has three keys:
        `constant_score`, `upper_bound` and `lower_bound`. The first is the
        constant score which lies between 0 and 1. The second is the upper
        bound for each channel that have been found to separate anomalies
        and normal points. The latter is the lower bound for each channel.
        If any of the bounds is None it means that there is no separation
        keeping TNR at 1. When bounds are not None, a point is labelled
        anomalous if any feature of that point is greater or lower than
        found bounds.
    """
    asc_x = np.sort(x, axis=0, kind="heapsort")
    desc_x = np.flip(asc_x, axis=0)
    
    def find_best_constants(channel: np.ndarray,
                            desc: np.ndarray,
                            asc: np.ndarray) -> tuple[float, float, float]:
        up, low = None, None
        # find best upper bound
        i = 0
        tnr, tpr, score = 1, 0, 0
        while tpr < 1 and tnr == 1 and i < desc.shape[0]:
            curr_pred = (channel >= desc[i]).reshape(-1)
            tpr = true_positive_rate(y, curr_pred)
            tnr = true_negative_rate(y, curr_pred)
            if tpr > score and tnr == 1:
                up = desc[i]
                score = tpr
            i += 1
        
        # find best lower bound
        tnr, tpr, score = 1, 0, 0
        i = 0
        while tpr < 1 and tnr == 1 and i < asc.shape[0]:
            curr_pred = (channel <= asc[i]).reshape(-1)
            tpr = true_positive_rate(y, curr_pred)
            tnr = true_negative_rate(y, curr_pred)
            if tpr > score and tnr == 1:
                low = asc[i]
                score = tpr
            i += 1
        return score, low, up
    
    # find the best constants and score feature-wise
    c_up = [None] * x.shape[1]
    c_low = [None] * x.shape[1]
    for f in range(x.shape[1]):
        _, c_low[f], c_up[f] = find_best_constants(x[:, f], desc_x[:, f], asc_x[:, f])
    
    # find the best score overall
    pred = np.zeros_like(y, dtype=bool)
    for f in range(x.shape[1]):
        if c_up[f] is not None and c_low[f] is not None:
            pred = pred | ((x[:, f] >= c_up[f]) | (x[:, f] <= c_low[f])).reshape(-1)
        elif c_up[f] is not None and c_low[f] is None:
            pred = pred | (x[:, f] >= c_up[f]).reshape(-1)
        elif c_up[f] is None and c_low[f] is not None:
            pred = pred | (x[:, f] <= c_low[f]).reshape(-1)
    pred = pred.reshape(-1)
    
    constant_score = true_positive_rate(y, pred)
    return {"constant_score": constant_score, "upper_bound": c_up, "lower_bound": c_low}


def _get_windows_to_try(window_range: tuple[int, int] | slice | list[int] = (2, 300)):
    """Converts window ranges into a list of windows to try.
    
    Parameters
    ----------
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try.

    Returns
    -------
    windows_to_try : list[int]
        The list of all windows to try given the window range.
    """
    if not isinstance(window_range, tuple) and not isinstance(window_range, slice) and not isinstance(window_range, list):
        raise TypeError("window_range must be a tuple, slice or list")
    
    if isinstance(window_range, tuple):
        if len(window_range) != 2:
            raise ValueError("window_range must be a tuple of two elements")
        elif window_range[0] > window_range[1]:
            raise ValueError("window_range[0] must be less or equal than "
                             f"window_range[1]. Received {window_range}")
        elif window_range[0] < 2:
            raise ValueError("window_range[0] must be at least 2")
    elif isinstance(window_range, list) and len(window_range) == 0:
        raise ValueError("window_range must have elements if it is a list")
    
    if isinstance(window_range, list):
        return window_range
    elif isinstance(window_range, slice):
        return [w for w in range(window_range.start, window_range.stop, window_range.step)]
    else:
        windows = []
        i = window_range[0]
        while i <= window_range[1]:
            windows.append(i)
        
            if i < 5:
                i += 1
            elif 5 <= i < 20:
                i += 5
            elif 20 <= i < 100:
                i += 10
            elif 100 <= i < 200:
                i += 20
            elif 200 <= i < 300:
                i += 50
            elif 300 <= i < 1000:
                i += 100
            else:
                i += math.floor(math.log10(i))
        
        if window_range[1] not in windows:
            windows.append(window_range[1])
        return windows


def analyse_constant_simplicity(x, y, diff: int = 3) -> dict:
    """Analyses whether the series is constant simple and its score.

    A dataset is constant simple if just by using one constant for each
    channel it is possible to separate normal and anomalous points. This
    function analyses this property and gives a score of 1 when the dataset
    is constant simple. It gives 0 when no anomalies can be found without
    producing false positives. Therefore, the higher the score, the higher
    the number of anomalies that can be found just by placing a constant.
    The score is the True Positive Rate (TPR) at True Negative Rate (TNR)
    equal to 1.

    The analysis tries to divide the normal and anomalous points just by
    placing a constant. Basically, it states that the anomalous and normal
    points can be perfectly divided one from the other.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.

    y : array-like of shape (n_samples,)
        The labels of the time series.

    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has four keys:
        `constant_score`, `upper_bound`, `lower_bound` and `diff_order`. The
        former is the constant score which lies between 0 and 1. The second is
        the upper bound for each channel that have been found to separate
        anomalies and normal points. The third is the lower bound for each
        channel. The latter is the differencing applied to the time series to
        find the constant score and constants. If any of the bounds is None it
        means that there is no separation keeping TNR at 1. When bounds are not
        None, a point is labelled anomalous if any feature of that point is
        greater or lower than found bounds.
    """
    if not isinstance(diff, int):
        raise TypeError("diff must be an integer")
    
    if diff < 0:
        raise ClosedOpenRangeError(0, np.inf, diff)
    
    check_array(x)
    check_X_y(x, y)
    x = np.array(x)
    y = np.array(y)
    
    best_result = None
    diffs = diff + 1
    for diff_order in range(diffs):
        series = x
        if diff_order != 0:
            series = np.diff(x, diff_order, axis=0)
        
        result = _find_constant_score(series, y[diff_order:])
        if best_result is None:
            best_result = result
            best_result["diff_order"] = diff_order
        elif result["constant_score"] > best_result["constant_score"]:
            best_result = result
            best_result["diff_order"] = diff_order
        
        if best_result["constant_score"] == 1:
            break
    
    return best_result


def _execute_movement_simplicity(x,
                                 y,
                                 diff: int,
                                 window_range: tuple[int, int] | slice | list[int],
                                 statistical_movement: Callable[[Any, int, Optional[str]], np.ndarray]) -> dict:
    """Computes the movement simplicity score.
    
    A movement is considered a statistical quantity computed over a sliding
    window on the original time series.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.
    
    statistical_movement : Callable[[Any, int, Optional[str]], np.ndarray]
        It is the function producing the time series of the statistical movement
        to be checked for simplicity.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has four keys:
        `movement_score`, `upper_bound`, `lower_bound` and `diff_order`. The
        former is the movement score which lies between 0 and 1. The second is
        the upper bound for each channel that have been found to separate
        anomalies and normal points. The third is the lower bound for each
        channel. The latter is the differencing applied to the time series to
        find the constant score and constants. If any of the bounds is None it
        means that there is no separation keeping TNR at 1. When bounds are not
        None, a point is labelled anomalous if any feature of that point is
        greater or lower than found bounds.
    """
    if not isinstance(diff, int):
        raise TypeError("diff must be an integer")
    elif not isinstance(statistical_movement, Callable):
        raise TypeError("statistical_movement must be Callable[[Any, int, str], np.ndarray]")
    
    if diff < 0:
        raise ClosedOpenRangeError(0, np.inf, diff)
    
    windows_to_try = _get_windows_to_try(window_range)
    
    check_array(x)
    check_X_y(x, y)
    x = np.array(x)
    y = np.array(y)
    
    best_result = None
    diffs = diff + 1
    for diff_order in range(diffs):
        series = x
        labels = y
        if diff_order != 0:
            series = np.diff(x, diff_order, axis=0)
            labels = y[diff_order:]
        
        for window in windows_to_try:
            movement = statistical_movement(series, window)
            result = analyse_constant_simplicity(movement, labels, 0)
            
            if best_result is None:
                best_result = result
                best_result["diff_order"] = diff_order
            elif result["constant_score"] > best_result["constant_score"]:
                best_result = result
                best_result["diff_order"] = diff_order
            
            if best_result["constant_score"] == 1:
                break
    
    best_result["movement_score"] = best_result["constant_score"]
    del best_result["constant_score"]
    return best_result


def analyse_mov_avg_simplicity(x,
                               y,
                               diff: int = 3,
                               window_range: tuple[int, int] | slice | list[int] = (2, 300)) -> dict:
    """Analyses whether the time series is moving average simple and its score.
    
    A dataset is moving average simple if just by placing a constant over
    the moving average series it is possible to separate normal and
    anomalous points. Here, the considered moving average can have window
    lengths all different for all channels. The function gives a sore of 1
    when the dataset is moving average simple. It gives 0 when no anomalies
    can be found without producing false positives. Therefore, the higher
    the score, the higher the number of anomalies that can be found just
    by placing a constant on the moving average series. The score is the
    True Positive Rate (TPR) at True Negative Rate (TNR) equal to 1.
    
    The analysis tries to divide the normal and anomalous points just by
    placing a constant in a moving average space of the time series. It
    means that the time series is first projected into a moving average
    space with window of length `w` to be found, then a constant to divide
    the points is found. If the time series is multivariate, the constant
    will be a constant vector in which elements are the constants for the
    time series channels and each channel may be projected with a different
    window `w`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has three keys:
        `mov_avg_score`, `upper_bound`, `lower_bound`, `window` and `diff_order`.
        The first is the moving average score which lies between 0 and 1. The
        second is the upper bound for each channel that have been found to
        separate anomalies and normal points. The third is the lower bound for
        each channel. The fourth is the best window that has been found to yield
        the score. The last is the differencing that has been applied to the
        time series before searching for the moving average score. If any of the
        bounds is None it means that there is no separation keeping TNR at 1.
        When bounds are not None, a point is labelled anomalous if any
        feature of the moving average series of that point is greater or
        lower than found bounds.
    """
    result = _execute_movement_simplicity(x, y, diff, window_range, mov_avg)
    result["moving_average_score"] = result["constant_score"]
    del result["constant_score"]
    return result


def analyse_mov_std_simplicity(x,
                               y,
                               diff: int = 3,
                               window_range: tuple[int, int] | slice | list[int] = (2, 300)) -> dict:
    """Analyses whether the time series is moving standard deviation simple and its score.
    
    A dataset is moving standard deviation simple if just by placing a
    constant over  the moving standard deviation series it is possible to
    separate normal and anomalous points. Here, the considered moving
    standard deviation can have window lengths all different for all
    channels. The function gives a sore of 1  when the dataset is moving
    standard deviation simple. It gives 0 when no anomalies can be found
    without producing false positives. Therefore, the higher the score, the
    higher the number of anomalies that can be found just by placing a
    constant on the moving average series. The score is the True Positive
    Rate (TPR) at True Negative Rate (TNR) equal to 1.
    
    The analysis tries to divide the normal and anomalous points just by
    placing a constant in a moving standard deviation space of the time
    series. It means that the time series is first projected into a moving
    standard deviation space with window of length `w` to be found, then a
    constant to divide the points is found. If the time series is
    multivariate, the constant will be a constant vector in which elements
    are the constants for the time series channels and each channel may be
    projected with a different window `w`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has three keys:
        `mov_std_score`, `upper_bound`, `lower_bound`, `window` and `diff_order`.
        The first is the moving standard deviation score which lies between 0
        and 1. The second is the upper bound for each channel that have been
        found to separate anomalies and normal points. The third is the lower
        bound for each channel. The fourth is the best window that has been
        found to yield the score. The last is the differencing that has been
        applied to the time series before searching for the moving standard
        deviation score. If any of the bounds is None it means that there is no
        separation keeping TNR at 1. When bounds are not None, a point is
        labelled anomalous if any feature of the moving average series of that
        point is greater or lower than found bounds.
    """
    result = _execute_movement_simplicity(x, y, diff, window_range, mov_std)
    result["moving_standard_deviation_score"] = result["constant_score"]
    del result["constant_score"]
    return result
