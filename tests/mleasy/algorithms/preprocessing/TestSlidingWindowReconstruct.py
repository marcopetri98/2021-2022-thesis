import itertools
import unittest

import numpy as np

from mleasy.algorithms.preprocessing import SlidingWindowReconstruct


class TestSlidingWindowReconstruct(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 6)
    
    def setUp(self) -> None:
        pass
    
    def test_shape_change(self):
        for window, stride, series in itertools.product([1, 2, 3, 7, 10], [1, 2, 3], [self.series_uni, self.series_multi]):
            print(f"Trying sliding window reconstruction with window={window}, stride={stride}")
            shape_changer = SlidingWindowReconstruct(window=window,
                                                     stride=stride)
            
            new_x, new_y = shape_changer.shape_change(series)
            self.assertEqual(series.ndim + 1, new_x.ndim)
            self.assertEqual(series.ndim + 1, new_y.ndim)
            self.assertEqual((series.shape[0] - window) // stride + 1, new_x.shape[0])
            self.assertEqual(window, new_x.shape[1])
            self.assertEqual(series.shape[1], new_x.shape[2])
            self.assertTupleEqual(new_x.shape, new_y.shape)
            
            for i in range(new_x.shape[0]):
                start = i * stride
                end = start + window
                np.testing.assert_array_equal(series[start:end], new_x[i])
