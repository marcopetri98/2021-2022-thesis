from typing import Union, Callable

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_array

class TimeSeriesAnomalyLOF(TimeSeriesAnomalyWindowWrapper, IParametric):	
    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01,
                 n_neighbors: int = 20,
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 metric: Union[str, Callable[[list, list], float]] = 'minkowski',
                 p: int = 2,
                 metric_params: dict = None,
                 contamination: Union[str, float] = 'auto',
                 novelty: bool = False,
                 n_jobs: int = None):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs

    def fit(self, x, y=None, *args, **kwargs) -> None:
        check_array(x)
        x = np.array(x)
        
        x_new, windows_per_point = self._project_time_series(x)
        self._build_wrapped()
        self._wrapped_model.fit(x_new)

    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        if self.novelty:
            check_not_default_attributes(self, {"_wrapped_model": None})
        return super().anomaly_score(x)
    
    def classify(self, X, *args, **kwargs) -> np.ndarray:
        if self.novelty:
            check_not_default_attributes(self, {"_wrapped_model": None})
        return super().classify(X)
    
    def _compute_window_labels(self, vector_data: np.ndarray) -> np.ndarray:
        # If the model is used as novelty it directly predicts
        if self.novelty:
            # Anomalies are -1 in LOF
            window_anomalies = self._wrapped_model.predict(vector_data) * -1
        else:
            self._build_wrapped()
            
            # Anomalies are -1 in LOF
            window_anomalies = self._wrapped_model.fit_predict(vector_data) * -1
            
        return window_anomalies

    def _compute_window_scores(self, vector_data: np.ndarray) -> np.ndarray:
        if self.novelty:
            window_scores = - self._wrapped_model.decision_function(vector_data)
        else:
            self._build_wrapped()
            
            # I use fit since I do not need the labels
            self._wrapped_model.fit(vector_data)
            window_scores = - self._wrapped_model.negative_outlier_factor_
        
        return window_scores

    def _build_wrapped(self) -> None:
        self._wrapped_model = LocalOutlierFactor(self.n_neighbors,
                                                 algorithm=self.algorithm,
                                                 leaf_size=self.leaf_size,
                                                 metric=self.metric,
                                                 p=self.p,
                                                 metric_params=self.metric_params,
                                                 contamination=self.contamination,
                                                 novelty=self.novelty,
                                                 n_jobs=self.n_jobs)