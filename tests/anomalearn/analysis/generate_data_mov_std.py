from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json


def reset_series_uni():
    return np.arange(100).reshape(-1, 1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "mov_std_simplicity"

# case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
case_num = 0
np.savetxt(str(this / f"mov_std_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_std_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_std_score": 0,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_std_case_{case_num}_result.json")

# case 1: score 1, diff 0, lower bound, window 2
# case 2: score 1, diff 0, upper bound, window 2
# case 3: score 1, diff 0, both bounds, window 2
# case 4: score 1, diff 1, lower bound, window 2
# case 5: score 1, diff 1, upper bound, window 2
# case 6: score 1, diff 1, both bounds, window 2
# case 7: 0 < score < 1, diff 0, lower bound, window 2
# case 8: 0 < score < 1, diff 0, upper bound, window 2
# case 9: 0 < score < 1, diff 0, both bounds, window 2
# case 10: 0 < score < 1, diff 1, lower bound, window 2
# case 11: 0 < score < 1, diff 1, upper bound, window 2
# case 12: 0 < score < 1, diff 1, both bounds, window 2
# case 13: score 1, diff 0, lower bound, window >2
# case 14: score 1, diff 0, upper bound, window >2
# case 15: score 1, diff 0, both bounds, window >2
# case 16: score 1, diff 1, lower bound, window >2
# case 17: score 1, diff 1, upper bound, window >2
# case 18: score 1, diff 1, both bounds, window >2
# case 19: 0 < score < 1, diff 0, lower bound, window >2
# case 20: 0 < score < 1, diff 0, upper bound, window >2
# case 21: 0 < score < 1, diff 0, both bounds, window >2
# case 22: 0 < score < 1, diff 1, lower bound, window >2
# case 23: 0 < score < 1, diff 1, upper bound, window >2
# case 24: 0 < score < 1, diff 1, both bounds, window >2
