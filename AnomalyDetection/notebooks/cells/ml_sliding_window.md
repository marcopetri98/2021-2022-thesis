# Approaches using sliding window in univariate time series

Univariate time series are characterized by a time ordered sequence of values, which in general are real numbers. Therefore, to be able to use spatial machine learning algorithms to perform anomaly detection, we need an approach to rearrange data to treat them as if they were spatial data. To do so, we define two parameters:

- Window: it is the number of consecutive samples that will be grouped together into a vector of window data. It must be at least 1.
- Stride: it is the number of steps by which we move the window. It must be at least 1.

Now, given that we have defined what sliding window methods works in general, we can even describe how anomalies are computed. If `stride < window`, several points will be used to create multiple windows, e.g., with `window = 3` and `stride = 1`, the point at index `i = 2` will be contained in three windows: the first, the second and the third. Therefore, once we compute the scores and the labels of windows, we need a way to come back to the time series space by scoring and labelling points of the time series. Because of that, we distinguish two ways to perform labelling of points and one way to compute the score of the points. The scoring methods are:

- Average: for each point we take all the windows containing this point. Then, we average the scores of the windows and we give this score to the point.

Moreover, score can be rescaled with the following options:

- None: scores are not normalized.
- MinMax normalization: scores are normalized using MinMax normalization implemented in `scikit-learn`.

The labelling methods are:

- Voting: given a threshold $\tau$ of agreement between windows built using a point $p$, if at least $\tau$ percentage of windows agree that the point is an anomaly, it will be classified as anomaly. Otherwise, the point is considered normal.
- Points score: given the scores of points, we compute the mean and standard deviation to fit a truncated gaussian of the scores. Then, each point having a score greater than the qth-q quantile specified is classified as anomaly. I.e., if we want to compute the qth-q quantile 0.999, every point whose probability is less than 0.001 will be classified as anomaly.