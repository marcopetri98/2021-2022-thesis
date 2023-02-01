import abc


class ITSADatasetSimplicity(abc.ABC):
    """An interface for objects analysing time series anomaly dataset's simplicity.
    """
    @abc.abstractmethod
    def analyse_constant_simplicity(self, x,
                                    y,
                                    verbose: bool = True,
                                    *args,
                                    **kwargs) -> dict:
        """Analyses whether the series is constant simple and its score.
        
        A dataset is constant simple if just by using one constant for each
        channel it is possible to separate normal and anomalous points. This
        function analyses this property and gives a score of 1 when the dataset
        is constant simple. It gives 0 when no anomalies can be found without
        producing false positives. Therefore, the higher the score, the higher
        the number of anomalies that can be found just by placing a constant.
        The score is the True Positive Rate (TPR) at True Negative Rate (TNR)
        equal to 1.
        
        The analysis tries to divide the normal and anomalous points just by
        placing a constant. Basically, it states that the anomalous and normal
        points can be perfectly divided one from the other.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The time series to be analysed.
            
        y : array-like of shape (n_samples,)
            The labels of the time series.

        verbose : bool, default=True
            Whether to print detailed information about constant simplicity
            computation.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        analysis_result : dict
            Dictionary with the results of the analysis. It has three keys:
            `constant_score`, `upper_bound` and `lower_bound`. The first is the
            constant score which lies between 0 and 1. The second is the upper
            bound for each channel that have been found to separate anomalies
            and normal points. The latter is the lower bound for each channel.
            If any of the bounds is None it means that there is no separation
            keeping TNR at 1. When bounds are not None, a point is labelled
            anomalous if any feature of that point is greater or lower than
            found bounds.
        """
    
    @abc.abstractmethod
    def analyse_mov_avg_simplicity(self, x,
                                   y,
                                   verbose: bool = True,
                                   *args,
                                   **kwargs) -> dict:
        """Analyses whether the time series is moving average simple and its score.
        
        A dataset is moving average simple if just by placing a constant over
        the moving average series it is possible to separate normal and
        anomalous points. Here, the considered moving average can have window
        lengths all different for all channels. The function gives a sore of 1
        when the dataset is moving average simple. It gives 0 when no anomalies
        can be found without producing false positives. Therefore, the higher
        the score, the higher the number of anomalies that can be found just
        by placing a constant on the moving average series. The score is the
        True Positive Rate (TPR) at True Negative Rate (TNR) equal to 1.
        
        The analysis tries to divide the normal and anomalous points just by
        placing a constant in a moving average space of the time series. It
        means that the time series is first projected into a moving average
        space with window of length `w` to be found, then a constant to divide
        the points is found. If the time series is multivariate, the constant
        will be a constant vector in which elements are the constants for the
        time series channels and each channel may be projected with a different
        window `w`.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The time series to be analysed.
            
        y : array-like of shape (n_samples,)
            The labels of the time series.

        verbose : bool, default=True
            Whether to print detailed information about moving average
            simplicity computation.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        analysis_result : dict
            Dictionary with the results of the analysis. It has three keys:
            `constant_score`, `upper_bound`, `lower_bound` and `window`. The
            first is the constant score which lies between 0 and 1. The second
            is the upper bound for each channel that have been found to separate
            anomalies and normal points. The third is the lower bound for each
            channel. If any of the bounds is None it means that there is no
            separation keeping TNR at 1. The latter is the window used for each
            channel, namely the series over which the constants are searched.
            When bounds are not None, a point is labelled anomalous if any
            feature of the moving average series of that point is greater or
            lower than found bounds.
        """
        pass
    
    @abc.abstractmethod
    def analyse_mov_std_simplicity(self, x,
                                   y,
                                   verbose: bool = True,
                                   *args,
                                   **kwargs) -> dict:
        """Analyses whether the time series is moving standard deviation simple and its score.
        
        A dataset is moving standard deviation simple if just by placing a
        constant over  the moving standard deviation series it is possible to
        separate normal and anomalous points. Here, the considered moving
        standard deviation can have window lengths all different for all
        channels. The function gives a sore of 1  when the dataset is moving
        standard deviation simple. It gives 0 when no anomalies can be found
        without producing false positives. Therefore, the higher the score, the
        higher the number of anomalies that can be found just by placing a
        constant on the moving average series. The score is the True Positive
        Rate (TPR) at True Negative Rate (TNR) equal to 1.
        
        The analysis tries to divide the normal and anomalous points just by
        placing a constant in a moving standard deviation space of the time
        series. It means that the time series is first projected into a moving
        standard deviation space with window of length `w` to be found, then a
        constant to divide the points is found. If the time series is
        multivariate, the constant will be a constant vector in which elements
        are the constants for the time series channels and each channel may be
        projected with a different window `w`.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The time series to be analysed.
            
        y : array-like of shape (n_samples,)
            The labels of the time series.

        verbose : bool, default=True
            Whether to print detailed information about moving standard
            deviation simplicity computation.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        analysis_result : dict
            Dictionary with the results of the analysis. It has three keys:
            `constant_score`, `upper_bound`, `lower_bound` and `window`. The
            first is the constant score which lies between 0 and 1. The second
            is the upper bound for each channel that have been found to separate
            anomalies and normal points. The third is the lower bound for each
            channel. If any of the bounds is None it means that there is no
            separation keeping TNR at 1. The latter is the window used for each
            channel, namely the series over which the constants are searched.
            When bounds are not None, a point is labelled anomalous if any
            feature of the moving standard deviation series of that point is
            greater or lower than found bounds.
        """
