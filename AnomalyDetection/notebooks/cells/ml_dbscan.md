# DBSCAN for time series

DBSCAN algorithm can easily find clusters and anomalies in data. In fact, in its standard implementation it computes some clusters without the need of specifying the number of them beforehand. Moreover, it also classifies points as outliers (anomalies). Therefore, it is straightforward that all the windows classified as anomalies by DBSCAN are indeed anomalies.

# Window scores

For what regards the scores, DBSCAN has not defined metrics to evaluate the score of a point either in terms of abnormality or in terms of normality. That being said, we need to find a way to compute scores of points. Currently, the score is computed as distance from the closest centroid. However, it may happen that non-globular shapes will bring several points (even anomalies) to be close to the clusters' centres. Therefore, it is important to find out a way to compute an anomaly score by directly exploiting density as it is defined by DBSCAN.