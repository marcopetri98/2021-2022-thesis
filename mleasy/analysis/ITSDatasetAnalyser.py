import abc
from abc import ABC
from typing import Tuple

import numpy as np

from . import IDatasetAnalyser, StationarityTest, DecompositionMethod


class ITSDatasetAnalyser(IDatasetAnalyser, ABC):
    """An interface for time series dataset analysers.
    """
    
    @abc.abstractmethod
    def analyse_stationarity(self, method: StationarityTest,
                             method_params: dict = None,
                             difference_series: bool = False,
                             difference_value: int = 1,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> None:
        """Analyse if the time series is stationary.
        
        Parameters
        ----------
        method : StationarityTest
            The method to be used for stationarity test.
            
        method_params : dict, default=None
            The additional parameters to pass to the stationarity test function.
        
        difference_series : bool, default=False
            If `True` the time series will be differenced, otherwise nothing
            will be done.
        
        difference_value : int, default=1
            It is the value defining how many times the time series must be
            differenced.
        
        verbose : bool, default=True
            If `True` detailed printing will be shown.
        
        args : list
            Left for inheritance freedom of adding parameters.
        
        kwargs : dict
            Left for inheritance freedom of adding parameters.

        Returns
        -------
        None
        """
        pass
    
    @abc.abstractmethod
    def show_acf_function(self, acf_params: dict = None,
                          difference_series: bool = False,
                          difference_value: int = 1,
                          fig_size: Tuple = (6, 6),
                          verbose: bool = True,
                          *args,
                          **kwargs) -> None:
        """Computes and show the ACF function.
        
        Parameters
        ----------
        acf_params : dict, default=None
            The additional parameters to pass to the acf function.
        
        difference_series : bool, default=False
            If `True` the time series will be differenced, otherwise nothing
            will be done.
        
        difference_value : int, default=1
            It is the value defining how many times the time series must be
            differenced.
            
        fig_size : Tuple, default=(12,12)
            The dimension of the figure to be shown.
        
        verbose : bool, default=True
            If `True` detailed printing will be shown.
        
        args : list
            Left for inheritance freedom of adding parameters.
        
        kwargs : dict
            Left for inheritance freedom of adding parameters.

        Returns
        -------
        None
        """
        pass
    
    @abc.abstractmethod
    def show_pacf_function(self, pacf_params: dict = None,
                           difference_series: bool = False,
                           difference_value: int = 1,
                           fig_size: Tuple = (6, 6),
                           verbose: bool = True,
                           *args,
                           **kwargs) -> None:
        """Computes and show PACF function.
        
        Parameters
        ----------
        pacf_params : dict, default=None
            The additional parameters to pass to the pacf function.
        
        difference_series : bool, default=False
            If `True` the time series will be differenced, otherwise nothing
            will be done.
        
        difference_value : int, default=1
            It is the value defining how many times the time series must be
            differenced.
            
        fig_size : Tuple, default=(12,12)
            The dimension of the figure to be shown.
        
        verbose : bool, default=True
            If `True` detailed printing will be shown.
        
        args : list
            Left for inheritance freedom of adding parameters.
        
        kwargs : dict
            Left for inheritance freedom of adding parameters.

        Returns
        -------
        None
        """
        
    @abc.abstractmethod
    def show_acf_pacf_functions(self, acf_params: dict,
                                pacf_params: dict,
                                difference_series: bool = False,
                                difference_value: int = 1,
                                fig_size: Tuple = (12, 12),
                                verbose: bool = True,
                                *args,
                                **kwargs) -> None:
        """Computes and show both ACF and PACF functions.
        
        Parameters
        ----------
        acf_params : dict, default=None
            The additional parameters to pass to the acf function.
        
        pacf_params : dict, default=None
            The additional parameters to pass to the pacf function.
        
        difference_series : bool, default=False
            If `True` the time series will be differenced, otherwise nothing
            will be done.
        
        difference_value : int, default=1
            It is the value defining how many times the time series must be
            differenced.
            
        fig_size : Tuple, default=(12,12)
            The dimension of the figure to be shown.
        
        verbose : bool, default=True
            If `True` detailed printing will be shown.
        
        args : list
            Left for inheritance freedom of adding parameters.
        
        kwargs : dict
            Left for inheritance freedom of adding parameters.

        Returns
        -------
        None
        """
    
    @abc.abstractmethod
    def decompose_time_series(self, method: DecompositionMethod,
                              method_params: dict = None,
                              difference_series: bool = False,
                              difference_value: int = 1,
                              verbose: bool = True,
                              *args,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose and show the decomposition results of the time series.
        
        Parameters
        ----------
        method : DecompositionMethod
            The decomposition method to be used to decompose the time series.
        
        method_params : dict, default=None
            The additional parameters to pass to the decomposition method.
        
        difference_series : bool, default=False
            If `True` the time series will be differenced, otherwise nothing
            will be done.
        
        difference_value : int, default=1
            It is the value defining how many times the time series must be
            differenced.
        
        verbose : bool, default=True
            If `True` detailed printing will be shown.
        
        args : list
            Left for inheritance freedom of adding parameters.
        
        kwargs : dict
            Left for inheritance freedom of adding parameters.

        Returns
        -------
        components : tuple
            Components of the decomposition. The first component is the
            trend-cycle component, the second is the seasonal component, and the
            third is the residual component.
        """
        pass
