from abc import ABC

from ... import IPredictor


class ITimeSeriesPredictor(IPredictor, ABC):
    """Interface for all models predicting a time series.
    
    This class is intended as a predictor for the points of the time series,
    not a labeller. All the classes implementing this method should provide
    either predictions n-step ahead predictions or reconstructions. This type
    of predictor states which are the prediction for a given time series.
    
    The predictions for a reconstruction models are sequences and for a
    forecaster are the n-step ahead predictions.
    """
