model = TimeSeriesAnomalyKMeans(window=3,
                                classification="voting",
                                anomaly_portion=0.0003,
                                anomaly_threshold=0.9888,
                                kmeans_params={"n_clusters": 4,
                                               "random_state": 22})

labels = model.classify(data_test.reshape((-1, 1)))
scores = model.anomaly_score(data_test.reshape((-1, 1)))