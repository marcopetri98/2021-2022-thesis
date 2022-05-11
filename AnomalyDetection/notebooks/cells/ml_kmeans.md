# Kmeans for time series

Kmeans has only three functions which has previously been specified by interfaces and, it does not need to add any other method to be able to compute labels and scores of windows.

To each function we pass the already transformed data as spatial form. Since Kmeans does not automatically compute the anomalies in data, we need both a way to compute the anomaly score of a point and whether the point is an anomaly or not. To do so, we introduce a patameter called `anomaly_portion` identifying a score over which points are considered anomalies. Therefore, we first compute scores and then labels.

## Window scores

Since every point at the end of Kmeans algorithm is assigned to the closest centroid, we compute the distance from the point and the cluster's centroid to which it is assigned. The scores will be exactly the distance between the point and its cluster's centroid.

## Window anomalies

Each window whose anomaly score is higher than the specified threshold is classified as being an anomalous window.