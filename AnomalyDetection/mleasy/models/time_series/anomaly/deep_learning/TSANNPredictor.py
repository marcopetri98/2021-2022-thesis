from typing import Tuple, Callable

import numpy as np
import tensorflow as tf

from mleasy.models.time_series.anomaly.deep_learning import TSANNStandard


class TSANNPredictor(TSANNStandard):
    """Class representing anomaly detection based on prediction.
    
    Please, note that for statistical approaches to compute the threshold, the
    stride must always be 1. Otherwise, there will be points that won't be
    predicted and the probability density function cannot be computed.
    
    This class implements the creation of input vectors for neural network
    models predicting one or multiple points in the future. It also implements
    the functions used to compute the error vectors to score each point where
    the higher the score the more anomalous.
    
    To compute the score the approach presented in Malhotra et al.
    (https://www.esann.org/proceedings/2015) is used. The error vector is
    error = [e11,e12,...,e1l,e21,...,e2l,...,ed1,...,edl] where eij is the error
    at time `t` of feature `i` predicted at time `t-j`. So we build the vector
    containing predictions [p11,p12,...,p1l,...] and the targets with the same
    shape. Then, we will call the method to compute the errors. Note that the
    targets are [p1,p1,...,p1 (l times),p2,p2,...].
    """

    def __init__(self, training_model: tf.keras.Model,
                 prediction_model: tf.keras.Model,
                 fitting_function: Callable[[tf.keras.Model,
                                             np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             int,
                                             list], tf.keras.callbacks.History],
                 prediction_horizon: int = 1,
                 validation_split: float = 0.1,
                 mean_cov_sets: str = "training",
                 threshold_sets: str = "training",
                 error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian",
                 *,
                 window: int = 200,
                 stride: int = 1,
                 batch_size: int = 32,
                 stateful_model: bool = False):
        super().__init__(training_model=training_model,
                         prediction_model=prediction_model,
                         fitting_function=fitting_function,
                         prediction_horizon=prediction_horizon,
                         validation_split=validation_split,
                         mean_cov_sets=mean_cov_sets,
                         threshold_sets=threshold_sets,
                         error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function,
                         window=window,
                         stride=stride,
                         batch_size=batch_size,
                         stateful_model=stateful_model)
    
    def _build_x_y_sequences(self, x: np.ndarray,
                             keep_batches: bool = False,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        samples = None
        targets = None

        # directly build the numpy arrays to improve speed
        for i in range(0, x.shape[0] - self.window - self.prediction_horizon + 1, self.stride):
            new_sample = np.transpose(np.array(x[i:i + self.window])).reshape((1, x.shape[1], self.window))
            new_target = np.transpose(np.array(x[i + self.window:i + self.window + self.prediction_horizon])).reshape((1, x.shape[1], self.prediction_horizon))

            if samples is None or targets is None:
                samples = new_sample
                targets = new_target
            else:
                samples = np.concatenate((samples, new_sample))
                targets = np.concatenate((targets, new_target))

        if keep_batches:
            samples, targets = self._remove_extra_points(samples=samples, targets=targets, verbose=verbose)

        return samples, targets
    
    def _fill_pred_targ_matrices(self, y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 mat_pred: np.ndarray,
                                 mat_true: np.ndarray) -> None:
        """Build target and prediction matrices.
        
        The function uses the fact that objects are passed by reference and
        automatically update the objects that are passed to it.
        
        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_features, horizon)
            It is the vector of the model's predictions.
        
        y_true : ndarray of shape (n_samples, n_features, horizon)
            It is the vector of the model's targets.
        
        mat_pred : ndarray of shape (n_points, n_features, horizon)
            It is the matrix containing the predictions for all the features
            at all horizons for all time instants.
        
        mat_true : ndarray of shape (n_points, n_features, horizon)
            It is the matrix containing the true values for all the features
            at all horizons for all time instants.

        Returns
        -------
        None
        """
        for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
            # get index of first and last predicted points for slicing
            start_idx = idx * self.stride + self.window
            end_idx = start_idx + self.prediction_horizon
            
            # fill the matrices with the predictions
            for f in range(pred.shape[0]):
                np.fill_diagonal(mat_pred[start_idx:end_idx, f, :], pred[f, :])
                np.fill_diagonal(mat_true[start_idx:end_idx, f, :], true[f, :])
