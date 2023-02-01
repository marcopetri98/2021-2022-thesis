import numpy as np
from sklearn.utils import check_array, check_X_y

from . import ITSADatasetSimplicity
from ..utils import true_positive_rate, true_negative_rate


class TSADatasetSimplicityAnalyser(ITSADatasetSimplicity):
    """Time series datasets' analyser for simplicity.
    
    The analyser looks for datasets properties describing if the dataset is
    simple and to which degree if so. If a dataset is not simple in these terms
    it does not mean it is hard or inherently hard. It only means that the
    normal and anomalous points can be easily divided in simple spaces or
    directly in the dataset's space.
    """
    def __init__(self):
        super().__init__()
    
    def analyse_constant_simplicity(self, x,
                                    y,
                                    verbose: bool = True,
                                    *args,
                                    **kwargs) -> dict:
        check_array(x)
        check_X_y(x, y)
        
        x = np.array(x)
        y = np.array(y)
        
        is_multivariate = x.shape[1] >= 2
        asc_x = np.sort(x, axis=0, kind="heapsort")
        desc_x = np.flip(asc_x, axis=0)
        c_up, c_low = None, None
        tnr, tpr, score = 1, 0, 0
        
        def find_best_constants(channel: np.ndarray,
                                desc: np.ndarray,
                                asc: np.ndarray) -> tuple[float, float, float]:
            nonlocal tpr, tnr, score
            up, low = None, None
            # find best upper bound
            i = 0
            tnr, tpr, score = 1, 0, 0
            while tpr < 1 and tnr == 1 and i < desc.shape[0]:
                curr_pred = (channel >= desc[i]).reshape(-1)
                tpr = true_positive_rate(y, curr_pred)
                tnr = true_negative_rate(y, curr_pred)
                if tpr > score:
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
                if tpr > score:
                    low = asc[i]
                    score = tpr
                i += 1
            return score, low, up
        
        if is_multivariate:
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
        else:
            # find the best constants and score
            _, c_low, c_up = find_best_constants(x, desc_x, asc_x)
            pred = np.zeros_like(y, dtype=bool)
            if c_up is not None and c_low is not None:
                pred = pred | ((x >= c_up) | (x <= c_low)).reshape(-1)
            elif c_up is not None and c_low is None:
                pred = pred | (x >= c_up).reshape(-1)
            elif c_up is None and c_low is not None:
                pred = pred | (x <= c_low).reshape(-1)
            pred = pred.reshape(-1)

        score = true_positive_rate(y, pred)
        return {"constant_score": score, "upper_bound": c_up, "lower_bound": c_low}
    
    def analyse_mov_avg_simplicity(self, x,
                                   y,
                                   verbose: bool = True,
                                   *args,
                                   **kwargs) -> dict:
        pass
    
    def analyse_mov_std_simplicity(self, x,
                                   y,
                                   verbose: bool = True,
                                   *args,
                                   **kwargs) -> dict:
        pass
