import unittest

import numpy as np

from anomalearn.analysis import analyse_constant_simplicity, analyse_mov_avg_simplicity, analyse_mov_std_simplicity


class TestDatasetSimplicityFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.uni_series = np.arange(100).reshape(-1, 1)
        self.uni_labels = np.zeros(self.uni_series.shape[0])
        self.multi_series = np.array([np.arange(100), np.arange(100)]).transpose()
        self.multi_labels = np.zeros(self.multi_series.shape[0])
        
    def test_analyse_constant_simplicity(self):
        for series, labels in zip([self.uni_series, self.multi_series], [self.uni_labels, self.multi_labels]):
            if series.shape[1] == 1:
                labels[20] = 1
            else:
                labels[20] = 1
                labels[35] = 1
            
            results = analyse_constant_simplicity(series, labels, diff=3)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(0, results["constant_score"])
            self.assertEqual(0, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([None], results["upper_bound"])
                self.assertListEqual([None], results["lower_bound"])
            else:
                self.assertListEqual([None, None], results["upper_bound"])
                self.assertListEqual([None, None], results["lower_bound"])

            if series.shape[1] == 1:
                series[20] = 1000
            else:
                series[20, 0] = 1000
                series[35, 1] = 750
            results = analyse_constant_simplicity(series, labels, diff=3)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            self.assertEqual(0, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([1000], results["upper_bound"])
                self.assertListEqual([None], results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([None, None], results["lower_bound"])
                
            if series.shape[1] == 1:
                series[20] = 21
            else:
                series[20, 0] = 21
                series[35, 1] = 37
            results = analyse_constant_simplicity(series, labels, diff=3)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            self.assertEqual(1, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([2], results["upper_bound"])
                self.assertListEqual([None], results["lower_bound"])
            else:
                self.assertListEqual([2, 3], results["upper_bound"])
                self.assertListEqual([None, None], results["lower_bound"])

            if series.shape[1] == 1:
                series[20] = 20
                labels[20] = 0
                series[70] = -1000
                labels[70] = 1
            else:
                series[20, 0] = -1000
                series[35, 1] = -750
            results = analyse_constant_simplicity(series, labels, diff=0)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            self.assertEqual(0, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([None], results["upper_bound"])
                self.assertListEqual([-1000], results["lower_bound"])
            else:
                self.assertListEqual([None, None], results["upper_bound"])
                self.assertListEqual([-1000, -750], results["lower_bound"])

            if series.shape[1] == 1:
                series[20] = 1000
                labels[20] = 1
            else:
                series[20, 0] = 1000
                series[35, 1] = 750
                series[20, 1] = -1000
                series[35, 0] = -750
            results = analyse_constant_simplicity(series, labels, diff=0)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            self.assertEqual(0, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([1000], results["upper_bound"])
                self.assertListEqual([-1000], results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([-750, -1000], results["lower_bound"])

            if series.shape[1] == 1:
                labels[50] = 1
            else:
                labels[50] = 1
            results = analyse_constant_simplicity(series, labels, diff=0)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
            self.assertEqual(2/3, results["constant_score"])
            self.assertEqual(0, results["diff_order"])
            if series.shape[1] == 1:
                self.assertListEqual([1000], results["upper_bound"])
                self.assertListEqual([-1000], results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([-750, -1000], results["lower_bound"])
    
    def test_analyse_mov_avg_simplicity(self):
        pass
    
    def test_analyse_mov_std_simplicity(self):
        pass
