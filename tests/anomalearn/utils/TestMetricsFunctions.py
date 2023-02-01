import unittest

from anomalearn.utils import true_positive_rate, true_negative_rate


class TestMetricsFunctions(unittest.TestCase):
    def test_true_positive_rate(self):
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(1, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(2/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(2/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(1/3, tpr)
        
        tpr = true_positive_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(0, tpr)
    
    def test_true_negative_rate(self):
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(1, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1])
        self.assertEqual(3/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1])
        self.assertEqual(3/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(2/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(1/4, tnr)
        
        tnr = true_negative_rate([0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(0, tnr)
