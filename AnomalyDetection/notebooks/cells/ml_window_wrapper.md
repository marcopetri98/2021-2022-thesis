# Wrapper of spatial models

Every machine learning spatial algorithm must be wrapped in some sense. We do not re-implement existing methods of scikit-learn or other. We wrap these methods using the Adapter design pattern and by transforming univariate time series data into spatial data by the previously cited projection of the univariate time series onto $\mathbb{R}^n$ where $n$ is the dimension of the window. At this point, we have that all models compute the anomaly score and labels with the same logic. Being that said, we implement these few methods which can again be shared by all implemented machine learning methods.