import numpy as np

from anomalearn.analysis import analyse_constant_simplicity
from anomalearn.analysis.dataset_simplicity import _find_constant_score, _execute_movement_simplicity
from anomalearn.utils import mov_avg

if __name__ == "__main__":
    method = None
    
    while method is None or method not in ["analyse", "find", "movement"]:
        method = input("Which function do you want to diagnose in terms of parallelism? [analyse/find/movement] ")

        dummy_series, dummy_labels = np.random.rand(100, 3), np.zeros(100)
        dummy_labels[50] = 1
        
        if method == "analyse":
            _ = analyse_constant_simplicity(dummy_series, dummy_labels, 1)
            analyse_constant_simplicity.parallel_diagnostics(level=4)
        elif method == "find":
            _ = _find_constant_score(dummy_series, dummy_labels)
            _find_constant_score.parallel_diagnostics(level=4)
        elif method == "movement":
            _ = _execute_movement_simplicity(dummy_series, dummy_labels, 1, (2, 3), mov_avg)
            _execute_movement_simplicity.parallel_diagnostics(level=4)
