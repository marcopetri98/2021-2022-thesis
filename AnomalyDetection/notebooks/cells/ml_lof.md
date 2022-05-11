# LOF for time series

Between all spatial method, LOF is the easiest approach to apply to univariate time series data. It is specifically desifned to perform anomaly detection. It is able to compute anomalies and to give an abnormality score of each point. Therefore, the only thing we need to do, is to call the wrapped LOF method and use the scores and the labels it computes for the windows.

# Unsupervised version

LOF allows also to be implemented as novelty detection, whose approach is self-supervised. It only needs to set the novelty flag to true. In such a case, LOF must be trained on a training set and then tested. It must not be used on the same set since it will have no meaning.