import abc

from . import IParametric


class IMultipleParametric(IParametric):
    """Interface identifying a machine learning parametric model with multiple fit.
    
    This class implements extends the standard IParametric interface to describe
    a model that can be trained on multiple datasets at the same time. E.g.,
    if we want to train on two different datasets the model we should use the
    function :meth:`models.IMultipleParametric.IMultipleParametric.fit_multiple`.
    Otherwise, we should learn the inherited method.
    """
    
    @abc.abstractmethod
    def fit_multiple(self, x: list, y: list = None, *args, **kwargs) -> None:
        """Fits the model to the given training data.
        
        Parameters
        ----------
        x : list
            The training data representing containing the features. It must be
            a list of array-like objects with shape (n_samples, n_features).
            Otherwise an exception will be triggered.

        y : list, default=None
            The target for the training data which may be used by either
            classification or regression models. It must be a list of array-like
            objects with shape (n_samples, n_features). Otherwise an exception
            will be triggered.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        pass
