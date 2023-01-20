import itertools
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from mleasy.algorithms.preprocessing import SlidingWindowForecast


class TestSlidingWindowForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 6)
        
    def test_equality(self):
        shape_changer1 = SlidingWindowForecast(window=10, stride=1, forecast=1)
        shape_changer2 = SlidingWindowForecast(window=10, stride=1, forecast=1)
        
        shape_changer3 = SlidingWindowForecast(window=5, stride=1, forecast=1)
        shape_changer4 = SlidingWindowForecast(window=10, stride=5, forecast=1)
        shape_changer5 = SlidingWindowForecast(window=10, stride=1, forecast=5)
        shape_changer6 = SlidingWindowForecast(window=5, stride=5, forecast=1)
        shape_changer7 = SlidingWindowForecast(window=5, stride=1, forecast=5)
        shape_changer8 = SlidingWindowForecast(window=10, stride=5, forecast=5)
        shape_changer9 = SlidingWindowForecast(window=5, stride=5, forecast=5)
        
        self.assertEqual(shape_changer1, shape_changer2)
        self.assertNotEqual(shape_changer1, shape_changer3)
        self.assertNotEqual(shape_changer1, shape_changer4)
        self.assertNotEqual(shape_changer1, shape_changer5)
        self.assertNotEqual(shape_changer1, shape_changer6)
        self.assertNotEqual(shape_changer1, shape_changer7)
        self.assertNotEqual(shape_changer1, shape_changer8)
        self.assertNotEqual(shape_changer1, shape_changer9)
        
        self.assertNotEqual(shape_changer1, 1866)
        self.assertNotEqual(shape_changer1, None)
        self.assertNotEqual(shape_changer1, "Sauron The Dark Lord")
        
    def test_copy(self):
        shape_changer = SlidingWindowForecast(window=15)
        new_obj = shape_changer.copy()
        
        self.assertEqual(shape_changer, new_obj)
        self.assertIsNot(shape_changer, new_obj)

    def test_save_and_load(self):
        shape_changer = SlidingWindowForecast(window=10, stride=2, forecast=2)
        
        self.assertIsNone(shape_changer.points_seen)
        self.assertEqual(10, shape_changer.window)
        self.assertEqual(2, shape_changer.stride)
        self.assertEqual(2, shape_changer.forecast)
        
        with TemporaryDirectory() as tmp_dir:
            shape_changer.save(tmp_dir)
            
            shape_changer = SlidingWindowForecast(window=-1, stride=-1, forecast=-1)
            shape_changer._points_seen = "Nazgul"
            shape_changer.load(tmp_dir)
            
            self.assertIsNone(shape_changer.points_seen)
            self.assertEqual(10, shape_changer.window)
            self.assertEqual(2, shape_changer.stride)
            self.assertEqual(2, shape_changer.forecast)
            
        shape_changer = SlidingWindowForecast(window=1, stride=1, forecast=1)
        shape_changer._points_seen = 10
        
        with TemporaryDirectory() as tmp_dir:
            shape_changer.save(tmp_dir)

            shape_changer = SlidingWindowForecast(window=-1, stride=-1, forecast=-1).load(tmp_dir)
            
            self.assertEqual(10, shape_changer.points_seen)
    
    def test_shape_change(self):
        for window, stride, forecast, series in itertools.product([1, 2, 3, 7, 10], [1, 2, 3], [1, 5, 7], [self.series_uni, self.series_multi]):
            print(f"Trying sliding window forecasting with window={window}, stride={stride}, forecast={forecast}")
            shape_changer = SlidingWindowForecast(window=window,
                                                  stride=stride,
                                                  forecast=forecast)
            
            new_x, new_y = shape_changer.shape_change(series)
            self.assertEqual(series.shape[0], shape_changer._points_seen)
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
