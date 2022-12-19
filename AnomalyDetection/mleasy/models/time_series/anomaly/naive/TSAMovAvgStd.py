import numpy as np
from scipy.optimize import brute
from sklearn.metrics import f1_score

from mleasy.models.time_series.anomaly.naive import TSAConstant
from mleasy.utils import print_header, print_step, mov_avg, mov_std


class TSAMovAvgStd(TSAConstant):
    """This class learns a moving average or moving standard deviation classifier.

    This class is a learner for the rule `y = 1(movxxx(series, w) > c)` or for
    the rule `y = 1(movxxx(series, w) < c)` where `1(param)` is the indicator
    function of param and `xxx` can either be `avg` or `std`. The rule can be
    learned either in a semi-supervised or in a supervised fashion.

    This model has two parameters, the window of the mov_avg or mov_std and the
    constant that must be used to decide whether a point is anomalous or not.
    Ones the derived series `movxxx(series, w)` has been computed, the constant
    `c` can be found by means of `TSAConstant` naive classifier.

    If self-supervised, the constant is learnt as the one for which all normal
    data are under or over it. If supervised, the constant and the window are
    learnt as the one maximizing the F1 on the training set.

    Parameters
    ----------
    comparison : ["less", "greater", "auto"], default="auto"
        The comparison that the constant must do. With "greater" the learned
        function is `y = 1(movavg(series, w) > c)`, with "less" it is
        `y = 1(movavg(series, w) < c)`. With "auto" and supervised  learning it
        chooses the better. Otherwise, it is identical to "greater".

    learning : ["semi-supervised", "supervised"], default="semi"
        States the type of "learning" that the function must perform. With
        "semi-supervised" it learns the constant from normal data only. With
        "supervised" it learns the constant from labeled data.
        
    max_window : int, default=100
        It is the maximum window of the time series to try to find the model.
        The model will search from windows within 3 and `max_window`.
        
    method : ["movavg", "movstd"], default="movavg"
        It is the moving window method to be used. It can either be "movavg" for
        moving average or "movstd" for moving standard deviation.

    Notes
    -----
    The class inherits from `TSAConstant` since it is conceptually the same
    model. It must learn a constant to compare with the values of movavg or
    mov_std to decide whether a point is anomalous. The only difference is that
    the constant is learnt over a moving average series and not on the original
    series.
    """
    def __init__(self, comparison: str = "auto",
                 learning: str = "semi-supervised",
                 max_window: int = 100,
                 method: str = "movavg"):
        super().__init__(comparison=comparison, learning=learning)

        self.max_window = max_window
        self.method = method

        self._window = 0
        self._mov_avg_series = None
        self._mov_std_series = None

    def get_window(self):
        return self._window

    def get_moving_series(self):
        return self._mov_avg_series if self._mov_avg_series is not None else self._mov_std_series

    def classify(self, x, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        self._mov_avg_series = None
        self._mov_std_series = None
        
        if verbose:
            print_header("Started samples' classification")
            print_step(f"Computing {self.method} series")
        
        if self.method == "movavg":
            mov_series = mov_avg(x, self._window)
            self._mov_avg_series = mov_series
        else:
            mov_series = mov_std(x, self._window)
            self._mov_std_series = mov_series
        
        result = super().classify(mov_series, verbose=verbose if verbose != 2 else True)
        
        if verbose:
            print_header("Ended samples' classification")
        
        return result

    def fit(self, x, y=None, verbose: bool = True, *args, **kwargs) -> None:
        x = np.array(x)
        
        if verbose:
            print_header("Started learning of window and constant")

        self._multivariate = x.ndim != 1 and x.shape[1] != 1

        def compute_1_minus_f1(window, *params):
            try:
                window = round(window[0])
            except Exception:
                window = round(window)

            if verbose:
                print_step(f"Trying window {window}")

            if window < 3 or window % 2 == 0:
                return 1
            else:
                half = int((window - 1) / 2)
                mov_series = mov_avg(x, window) if self.method == "movavg" else mov_std(x, window)
                targets = y[half:-half]
                super(TSAMovAvgStd, self).fit(mov_series, targets, verbose=verbose if verbose != 2 else True)
                
                return 1 - f1_score(targets, super(TSAMovAvgStd, self).classify(mov_series, verbose=False))

        optimal_window = brute(compute_1_minus_f1, [(3, self.max_window)])
        self._window = round(optimal_window[0])
        _ = compute_1_minus_f1(self._window)
        
        if verbose:
            print_header("Ended learning of window and constant")
