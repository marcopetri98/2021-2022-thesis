from copy import deepcopy
from typing import Tuple

import numpy as np
from statsmodels.tsa.seasonal import STL, DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from . import ITSDatasetAnalyser, StationarityTest, DecompositionMethod
from ..utils.printing import print_header, print_step
from ..visualizer.time_series import plot_correlation_functions, plot_time_series_decomposition


class TSDatasetAnalyser(ITSDatasetAnalyser):
    """Concrete class for time series dataset analysis.
    
    Parameters
    ----------
    time_series : ndarray
        It is the time series over which the methods can be performed.
    """
    def __init__(self, time_series: np.ndarray):
        super().__init__()
        
        if time_series.ndim != 1:
            raise ValueError("Time series must be mono-dimensional")
        elif time_series.shape[0] < 2:
            raise ValueError("Time series must have at least 2 values")
        
        self.time_series = deepcopy(time_series)
        
    def num_samples(self) -> int:
        return self.time_series.shape[0]
        
    def analyse_stationarity(self, method: StationarityTest,
                             method_params: dict = None,
                             difference_series: bool = False,
                             difference_value: int = 1,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> None:
        print_header("Stationarity analysis")
        
        if method_params is None:
            method_params = {}
        
        # difference the series if necessary
        analysed_series = self.time_series
        if difference_series:
            analysed_series = np.diff(analysed_series, difference_value)
            
        if verbose:
            print_step("Start stationarity test")
            
        match method:
            case StationarityTest.ADFULLER:
                test, p_value, _, _, _, _ = adfuller(analysed_series,
                                                     **method_params)
            
            case StationarityTest.KPSS:
                test, p_value, _, _ = kpss(analysed_series,
                                           **method_params)
            
            case _:
                raise NotImplementedError("{} not supported".format(method))
            
        print_step("Statistical test of {} is: {}".format(method, test))
        print_step("Computed p-value of {} is: {}".format(method, p_value))
            
        print_header("Stationarity analysis ended")
    
    def show_acf_function(self, acf_params: dict = None,
                          difference_series: bool = False,
                          difference_value: int = 1,
                          fig_size: Tuple = (6, 6),
                          verbose: bool = True,
                          *args,
                          **kwargs) -> None:
        print_header("ACF computation")
        
        if acf_params is None:
            acf_params = {}
    
        # difference the series if necessary
        analysed_series = self.time_series
        if difference_series:
            analysed_series = np.diff(analysed_series, difference_value)
            
        series_acf, series_conf = acf(analysed_series, **acf_params)
        plot_correlation_functions({"PACF": series_acf},
                                   {"PACF": series_conf},
                                   fig_size=fig_size)
        
        print_header("ACF computation ended")
    
    def show_pacf_function(self, pacf_params: dict = None,
                           difference_series: bool = False,
                           difference_value: int = 1,
                           fig_size: Tuple = (6, 6),
                           verbose: bool = True,
                           *args,
                           **kwargs) -> None:
        print_header("PACF computation")
        
        if pacf_params is None:
            pacf_params = {}
    
        # difference the series if necessary
        analysed_series = self.time_series
        if difference_series:
            analysed_series = np.diff(analysed_series, difference_value)
    
        series_pacf, series_conf = pacf(analysed_series, **pacf_params)
        plot_correlation_functions({"PACF": series_pacf},
                                   {"PACF": series_conf},
                                   fig_size=fig_size)
    
        print_header("PACF computation ended")
    
    def show_acf_pacf_functions(self, acf_params: dict,
                                pacf_params: dict,
                                difference_series: bool = False,
                                difference_value: int = 1,
                                fig_size: Tuple = (12, 12),
                                verbose: bool = True,
                                *args,
                                **kwargs) -> None:
        print_header("ACF and PACF computation")
        
        if acf_params is None:
            acf_params = {}
        if pacf_params is None:
            pacf_params = {}
    
        # difference the series if necessary
        analysed_series = self.time_series
        if difference_series:
            analysed_series = np.diff(analysed_series, difference_value)
    
        series_acf, series_acf_conf = acf(analysed_series, **acf_params)
        series_pacf, series_pacf_conf = pacf(analysed_series, **pacf_params)
        plot_correlation_functions({"ACF": series_acf, "PACF": series_pacf},
                                   {"ACF": series_acf_conf, "PACF": series_pacf_conf},
                                   fig_size=fig_size)
    
        print_header("ACF and PACF computation ended")
    
    def decompose_time_series(self, method: DecompositionMethod,
                              method_params: dict = None,
                              difference_series: bool = False,
                              difference_value: int = 1,
                              verbose: bool = True,
                              x_ticks_loc=None,
                              x_ticks_labels=None,
                              x_ticks_rotation: float = 0,
                              *args,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        x_ticks_loc : array-like, default=None
        The location at which printing the ticks labels. These will be also the
        labels in case the argument `x_ticks_labels` is None.

        x_ticks_labels : array-like, default=None
            The labels of the ticks on the x to be printed on the plot, they start
            at the first sample and end at the last sample if `x_ticks_loc` is None.
            Otherwise, they will be printed exactly at the position specified by the
            other argument.
            
        x_ticks_rotation : float, default=0.0
            The rotation of the ticks on the x-axis.
        """
        print_header("Series decomposition")
        
        if method_params is None:
            method_params = {}

        # difference the series if necessary
        analysed_series = self.time_series
        if difference_series:
            analysed_series = np.diff(analysed_series, difference_value)
        
        match method:
            case DecompositionMethod.STL:
                stl = STL(analysed_series, **method_params)
                res: DecomposeResult = stl.fit()
                original, seasonal, trend_cycle, residual = res.observed, res.seasonal, res.trend, res.resid
                
            case DecompositionMethod.MOVING_AVERAGE:
                res: DecomposeResult = seasonal_decompose(analysed_series, **method_params)
                original, seasonal, trend_cycle, residual = res.observed, res.seasonal, res.trend, res.resid
                
            case _:
                raise NotImplementedError("{} not supported".format(method))

        plot_time_series_decomposition(original,
                                       seasonal,
                                       trend_cycle,
                                       residual,
                                       x_ticks_loc=x_ticks_loc,
                                       x_ticks_labels=x_ticks_labels,
                                       x_ticks_rotation=x_ticks_rotation)
        
        print_header("Decomposition ended")
        
        return trend_cycle, seasonal, residual
