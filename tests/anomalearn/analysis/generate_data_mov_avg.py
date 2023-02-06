from pathlib import Path

import numpy as np

from anomalearn.utils import save_py_json, load_py_json


def reset_series_uni():
    return np.arange(100).reshape(-1, 1), np.zeros(100)


this = Path(__file__).parent / "test_data" / "mov_avg_simplicity"

# case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
case_num = 0
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0,
              "diff_order": 0,
              "lower_bound": [None],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 1: score 1, diff 0, lower bound, window 2
# case 2: score 1, diff 0, upper bound, window 2
# case 3: score 1, diff 0, both bounds, window 2
# case 4: score 1, diff 1, lower bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 0
uni_labels[20] = 1
case_num = 4
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [None],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 5: score 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_series[20] = 40
uni_labels[20] = 1
case_num = 5
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [11],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 6: score 1, diff 1, both bounds, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[35] = 1
uni_labels[20] = 1
uni_series[20] = 0
uni_series[35] = 65
uni_series[36] = 60
uni_series[37] = 55
uni_series[38] = 50
uni_series[39] = 45
case_num = 6
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 1,
              "diff_order": 1,
              "lower_bound": [-9],
              "upper_bound": [16],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

# case 7: 0 < score < 1, diff 0, lower bound, window 2
# case 8: 0 < score < 1, diff 0, upper bound, window 2
# case 9: 0 < score < 1, diff 0, both bounds, window 2
# case 10: 0 < score < 1, diff 1, lower bound, window 2
# case 11: 0 < score < 1, diff 1, upper bound, window 2
uni_series, uni_labels = reset_series_uni()
uni_labels[20] = 1
uni_labels[35] = 1
uni_series[35] = 65
case_num = 11
np.savetxt(str(this / f"mov_avg_case_{case_num}.csv"), uni_series, delimiter=",")
np.savetxt(str(this / f"mov_avg_case_{case_num}_labels.csv"), uni_labels, delimiter=",")
save_py_json({"mov_avg_score": 0.5,
              "diff_order": 1,
              "lower_bound": [None],
              "upper_bound": [16],
              "window": 2},
             this / f"mov_avg_case_{case_num}_result.json")

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
