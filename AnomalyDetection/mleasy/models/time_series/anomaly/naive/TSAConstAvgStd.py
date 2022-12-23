import numpy as np
from scipy.optimize import brute
from sklearn.metrics import f1_score

from mleasy.models import IAnomalyClassifier, IParametric
from mleasy.utils import mov_avg, mov_std, print_header, print_step


class TSAConstAvgStd(IAnomalyClassifier, IParametric):
    """This class learns a simple classifier.
    
    This class is a learner for the rule `y = 1(a * movavg(x, w) + b * movstd(x, w)
    + c < x) OR 1(a * movavg(x, w) - b * movstd(x, w) + c > x)` where `w` is an
    odd number representing the moving window, `movavg` stands for moving
    average, `movstd` stands for moving standard deviation, and `a`, `b`, `c`
    are real numbers. The symbol `x` stands for the input time series. The
    function `1(param)` is the indicator function. The rule can be learned
    either in a semi-supervised fashion or in a fully supervised fashion.
    
    If self-supervised, the parameters are learnt as the ones for which all
    normal data are under it. If supervised, the constant is learned as the one
    maximizing the F1 on the training set.
    
    Parameters
    ----------
    learning : ["supervised"], default="supervised"
        States the type of "learning" that the function must perform. With
        "semi-supervised" it learns the constant from normal data only. With
        "supervised" it learns the constant from labeled data.
        
    max_window : int, default=100
        It is the maximum window of the time series to try to find the model.
        The model will search from windows within 3 and `max_window`.
    """
    def __init__(self, learning: str = "supervised",
                 max_window: int = 100):
        super().__init__()
        
        self.learning = learning
        self.max_window = max_window

        self._anomaly_label = 1
        self._multivariate = None
        self._lower_series = None
        self._upper_series = None
        self._a = None
        self._b = None
        self._c = None
        self._w = None
        
    def get_a(self):
        return self._a
        
    def get_b(self):
        return self._b
        
    def get_c(self):
        return self._c
        
    def get_w(self):
        return self._w
        
    def get_upper_series(self):
        return self._upper_series
        
    def get_lower_series(self):
        return self._lower_series

    def classify(self, x, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True detailed printing is done.
        """
        if verbose:
            print_header("Started samples' classification")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if verbose:
            print_step("Computing the moving average and moving std series")

        moving_avg = mov_avg(x, self._w)
        moving_std = mov_std(x, self._w)
        half = int((self._w - 1) / 2)
        self._upper_series = self._a * moving_avg + self._b * moving_std + self._c
        self._lower_series = self._a * moving_avg - self._b * moving_std + self._c

        if self._upper_series.ndim == 1:
            self._upper_series = self._upper_series.reshape(-1, 1)
            self._lower_series = self._lower_series.reshape(-1, 1)

        if verbose:
            print_step("Building the predictions")

        pred = (x[half:-half] > self._upper_series) | (x[half:-half] < self._lower_series)
        middle = np.array(list(map(lambda row: 1 if np.max(row) == 1 else 0, pred)))
        all_predictions = np.full(x.shape[0], np.nan)
        all_predictions[half:-half] = middle

        if verbose:
            print_header("Ended samples' classification")

        return all_predictions
    
    def fit(self, x, y=None, verbose: bool = True, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        verbose : bool or int, default=True
            States if verbose printing must be done. With False very little
            printing is performed. With True detailed printing is done.
        """
        x = np.array(x)
        y = np.array(y)

        if verbose:
            print_header("Started learning")
            print_step("Checking if the time series is multivariate")

        self._multivariate = x.ndim != 1 and x.shape[1] != 1

        def compute_1_minus_f1(points, *params):
            a, b, c, w = points
            series, targets = params

            w = round(w)
            if w % 2 == 0:
                return 1
            
            moving_avg = mov_avg(series, w)
            moving_std = mov_std(series, w)
            half = int((w - 1) / 2)
            
            upper_boundary = a * moving_avg + b * moving_std + c
            lower_boundary = a * moving_avg - b * moving_std + c
            series = series[half:-half]
            
            pred = (series > upper_boundary.reshape(-1, 1)) | (series < lower_boundary.reshape(-1, 1))
            return 1 - f1_score(targets[half:-half], pred)

        def globally_optimize(extra_args):
            return brute(compute_1_minus_f1,
                         (slice(0, 2, 1), slice(-4, 4.1, 0.1), (np.min(extra_args[0]), np.max(extra_args[0])), (3, self.max_window)),
                         args=extra_args)

        if self.learning == "supervised":
            y = np.array(y)
            
            if y.shape[0] != x.shape[0]:
                raise ValueError("x and y must have the same number of points")
            elif self._anomaly_label not in y:
                raise ValueError("supervised training requires at least one anomaly")
            
            if self._multivariate:
                if verbose:
                    print_step("Getting the optimal parameters for each channel")

                self._a = np.zeros(x.shape[1])
                self._b = np.zeros(x.shape[1])
                self._c = np.zeros(x.shape[1])
                self._w = np.zeros(x.shape[1])
                for i in range(x.shape[1]):
                    extra_params = (x[:, i], y)
                    optimum = globally_optimize(extra_params)
                    self._a[i], self._b[i], self._c[i], self._w[i] = optimum
                    self._w[i] = round(self._w[i])
            else:
                if verbose:
                    print_step("Getting the optimal parameters")

                extra_params = (x, y)
                optimum = globally_optimize(extra_params)
                self._a, self._b, self._c, self._w = optimum
                self._w = round(self._w)

        if verbose:
            print_header("Ended learning")

    def __check_parameters(self):
        if not isinstance(self.learning, str):
            raise TypeError("learning must be a str")

        learnings = ["supervised"]

        if self.learning not in learnings:
            raise ValueError(f"learning must be one of {learnings}")
