import unittest

import numpy as np

from anomalearn.analysis import TSADatasetSimplicityAnalyser


class TestTSADatasetSimplicityAnalyser(unittest.TestCase):
    def setUp(self) -> None:
        self.uni_series = np.arange(100).reshape(-1, 1)
        self.uni_labels = np.zeros(self.uni_series.shape[0])
        self.multi_series = np.array([np.arange(100), np.arange(100)]).transpose()
        self.multi_labels = np.zeros(self.multi_series.shape[0])
        
    def test_analyse_constant_simplicity(self):
        analyser = TSADatasetSimplicityAnalyser()
        
        for series, labels in zip([self.uni_series, self.multi_series], [self.uni_labels, self.multi_labels]):
            if series.shape[1] == 1:
                labels[20] = 1
            else:
                labels[20] = 1
                labels[35] = 1
            
            results = analyser.analyse_constant_simplicity(series, labels)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound"}, set(results.keys()))
            self.assertEqual(0, results["constant_score"])
            if series.shape[1] == 1:
                self.assertIsNone(results["upper_bound"])
                self.assertIsNone(results["lower_bound"])
            else:
                self.assertListEqual([None, None], results["upper_bound"])
                self.assertListEqual([None, None], results["lower_bound"])

            if series.shape[1] == 1:
                series[20] = 1000
            else:
                series[20, 0] = 1000
                series[35, 1] = 750
            results = analyser.analyse_constant_simplicity(series, labels)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            if series.shape[1] == 1:
                self.assertEqual(1000, results["upper_bound"])
                self.assertIsNone(results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([None, None], results["lower_bound"])

            if series.shape[1] == 1:
                series[20] = 19
                labels[20] = 0
                series[70] = -1000
                labels[70] = 1
            else:
                series[20, 0] = -1000
                series[35, 1] = -750
            results = analyser.analyse_constant_simplicity(series, labels)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            if series.shape[1] == 1:
                self.assertIsNone(results["upper_bound"])
                self.assertEqual(-1000, results["lower_bound"])
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
            results = analyser.analyse_constant_simplicity(series, labels)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound"}, set(results.keys()))
            self.assertEqual(1, results["constant_score"])
            if series.shape[1] == 1:
                self.assertEqual(1000, results["upper_bound"])
                self.assertEqual(-1000, results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([-750, -1000], results["lower_bound"])

            if series.shape[1] == 1:
                labels[50] = 1
            else:
                labels[50] = 1
            results = analyser.analyse_constant_simplicity(series, labels)
            self.assertSetEqual({"constant_score", "upper_bound", "lower_bound"}, set(results.keys()))
            self.assertEqual(2/3, results["constant_score"])
            if series.shape[1] == 1:
                self.assertEqual(1000, results["upper_bound"])
                self.assertEqual(-1000, results["lower_bound"])
            else:
                self.assertListEqual([1000, 750], results["upper_bound"])
                self.assertListEqual([-750, -1000], results["lower_bound"])
    
    def test_analyse_mov_avg_simplicity(self):
        pass
    
    def test_analyse_mov_std_simplicity(self):
        pass
