import time
import unittest
from pathlib import Path

import numpy as np

from anomalearn.analysis import analyse_constant_simplicity, analyse_mov_avg_simplicity, analyse_mov_std_simplicity
from anomalearn.reader.time_series import SMDReader
from anomalearn.utils import load_py_json


def key_order(x):
    try:
        return int(x.name.split(".")[0].split("_")[-1])
    except ValueError:
        return int(x.name.split(".")[0].split("_")[-2])


class TestDatasetSimplicityFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.uni_series = np.arange(100).reshape(-1, 1)
        self.uni_labels = np.zeros(self.uni_series.shape[0])
        self.multi_series = np.array([np.arange(100), np.arange(100)]).transpose()
        self.multi_labels = np.zeros(self.multi_series.shape[0])
        
    def test_analyse_constant_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None)
        # case 1: score 1, diff 0, lower bound
        # case 2: score 1, diff 0, upper bound
        # case 3: score 1, diff 0, both bounds
        # case 4: score 1, diff 1, lower bound
        # case 5: score 1, diff 1, upper bound
        # case 6: score 1, diff 1, both bounds
        # case 7: 0 < score < 1, diff 0, lower bound
        # case 8: 0 < score < 1, diff 0, upper bound
        # case 9: 0 < score < 1, diff 0, both bounds
        # case 10: 0 < score < 1, diff 1, lower bound
        # case 11: 0 < score < 1, diff 1, upper bound
        # case 12: 0 < score < 1, diff 1, both bounds
        def assert_results(results_, exp_results_):
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results_.keys()))
            self.assertEqual(exp_results_["constant_score"], results_["constant_score"])
            self.assertEqual(exp_results_["diff_order"], results_["diff_order"])
            self.assertListEqual(exp_results_["upper_bound"], results_["upper_bound"])
            self.assertListEqual(exp_results_["lower_bound"], results_["lower_bound"])
        
        test_data = Path(__file__).parent / "test_data" / "constant_simplicity"
        cases = sorted([e for e in test_data.glob("const_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("const_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("const_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_constant_simplicity(series, labels, diff=3)
            assert_results(results, exp_results)
    
    def test_analyse_mov_avg_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
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
        def assert_results(results_, exp_results_):
            self.assertSetEqual({"mov_avg_score", "upper_bound", "lower_bound", "diff_order", "window"}, set(results_.keys()))
            self.assertEqual(exp_results_["mov_avg_score"], results_["mov_avg_score"])
            self.assertEqual(exp_results_["diff_order"], results_["diff_order"])
            self.assertEqual(exp_results_["window"], results_["window"])
            self.assertListEqual(exp_results_["lower_bound"], results_["lower_bound"])
            self.assertListEqual(exp_results_["upper_bound"], results_["upper_bound"])
        
        test_data = Path(__file__).parent / "test_data" / "mov_avg_simplicity"
        cases = sorted([e for e in test_data.glob("mov_avg_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_mov_avg_simplicity(series, labels, window_range=(2, 100), diff=3)
            assert_results(results, exp_results)
    
    def test_analyse_mov_std_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
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
        def assert_results(results_, exp_results_):
            self.assertSetEqual({"mov_std_score", "upper_bound", "lower_bound", "diff_order", "window"}, set(results_.keys()))
            self.assertEqual(exp_results_["mov_std_score"], results_["mov_std_score"])
            self.assertEqual(exp_results_["diff_order"], results_["diff_order"])
            self.assertEqual(exp_results_["window"], results_["window"])
            self.assertListEqual(exp_results_["lower_bound"], results_["lower_bound"])
            self.assertListEqual(exp_results_["upper_bound"], results_["upper_bound"])
        
        test_data = Path(__file__).parent / "test_data" / "mov_std_simplicity"
        cases = sorted([e for e in test_data.glob("mov_std_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("mov_std_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("mov_std_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_mov_std_simplicity(series, labels, window_range=(2, 100), diff=3)
            assert_results(results, exp_results)

    def test_speed(self):
        standard_path = Path(__file__).parent / "../../../data/anomaly_detection/smd"
        reader = SMDReader(str(standard_path))
        big_series = reader.read("machine-1-1", verbose=False).get_dataframe()
        values = big_series[sorted(set(big_series.columns).difference(["class", "timestamp", "is_training", "interpretation"]),
                                   key=lambda x: int(x.split("_")[-1]))].values
        labels = big_series["class"].values

        print(f"Start to analyse constant simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_constant_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")

        print(f"Start to analyse moving average simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_mov_avg_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")

        print(f"Start to analyse moving standard deviation simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_mov_std_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")
