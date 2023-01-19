import itertools
import unittest

import numpy as np

from mleasy.algorithms.preprocessing import SlidingWindowForecast


class TestSlidingWindowForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 6)
    
    def setUp(self) -> None:
        pass
    
    def test_shape_change(self):
        for window, stride, forecast, series in itertools.product([1, 2, 3, 7, 10], [1, 2, 3], [1, 2, 5, 7, 10], [self.series_uni, self.series_multi]):
            print(f"Trying sliding window reconstruction with window={window}, stride={stride}, forecast={forecast}")
            shape_changer = SlidingWindowForecast(window=window,
                                                  stride=stride,
                                                  forecast=forecast)
            
            new_x, new_y = shape_changer.shape_change(series)
            self.assertEqual(series.ndim + 1, new_x.ndim)
            self.assertEqual(series.ndim + 1, new_y.ndim)
            self.assertEqual((series.shape[0] - window - forecast) // stride + 1, new_x.shape[0])
            self.assertEqual(window, new_x.shape[1])
            self.assertEqual(series.shape[1], new_x.shape[2])
            self.assertEqual(forecast, new_y.shape[1])
            self.assertEqual(series.shape[1], new_y.shape[2])
            
            for i in range(new_x.shape[0]):
                start = i * stride
                end = start + window
                np.testing.assert_array_equal(series[start:end], new_x[i])
            
            for i in range(new_y.shape[0]):
                start = i * stride + window
                end = start + forecast
                np.testing.assert_array_equal(series[start:end], new_y[i])
