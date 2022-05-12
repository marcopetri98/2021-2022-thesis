# Get labels and scores for points

To be able to get scores and labels with unsupervised approach, given all the previous architecture, we can simply declare the model and call the `classify` function to get the labels for each point and we call `anomaly_score` to get the anomaly score of each point.