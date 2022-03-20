# Python imports
from typing import Union, Callable

# External imports
import numpy as np
from sklearn.base import OutlierMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

# Project imports
from models.transformers.TimeSeriesProjector import TimeSeriesProjector


class TimeSeriesAnomalyLOF(LocalOutlierFactor, OutlierMixin):
	"""LOF adapter for time series.

	It is a wrapper of the scikit-learn LOF approach. It uses the
	TimeSeriesProjector to project the time series onto a vector space. Then,
	it uses LOF to find all the anomalies and compute the score of an anomaly
	as described in the fit_predict method.
	
	Parameters
	----------
	window : int
		The length of the window to consider performing anomaly detection.
		
	stride : int
		The offset at which the window is moved when computing the anomalies.
	
	classification: {"voting", "points_score"}, default="voting"
		It defines the way in which a point is declared as anomaly. With voting,
		a point is an anomaly if at least anomaly_threshold percentage of
		windows containing the point agree in saying it is an anomaly. With
		points_score, the points are considered anomalies if they're score is
		above anomaly_threshold.
	
	anomaly_threshold: float, default=0.0
		The threshold used to compute if a point is an anomaly or not.
	
	n_neighbors : int, default=20
		Number of neighbors to use by default for :meth:`kneighbors` queries.
		If n_neighbors is larger than the number of samples provided,
		all samples will be used.

	algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
		Algorithm used to compute the nearest neighbors:

		- 'ball_tree' will use :class:`BallTree`
		- 'kd_tree' will use :class:`KDTree`
		- 'brute' will use a brute-force search.
		- 'auto' will attempt to decide the most appropriate algorithm
		  based on the values passed to :meth:`fit` method.

		Note: fitting on sparse input will override the setting of
		this parameter, using brute force.

	leaf_size : int, default=30
		Leaf is size passed to :class:`BallTree` or :class:`KDTree`. This can
		affect the speed of the construction and query, as well as the memory
		required to store the tree. The optimal value depends on the
		nature of the problem.

	metric : str or callable, default='minkowski'
		The metric is used for distance computation. Any metric from scikit-learn
		or scipy.spatial.distance can be used.

		If metric is "precomputed", X is assumed to be a distance matrix and
		must be square. X may be a sparse matrix, in which case only "nonzero"
		elements may be considered neighbors.

		If metric is a callable function, it is called on each
		pair of instances (rows) and the resulting value recorded. The callable
		should take two arrays as input and return one value indicating the
		distance between them. This works for Scipy's metrics, but is less
		efficient than passing the metric name as a string.

		Valid values for metric are:

		- from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
		  'manhattan']

		- from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
		  'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
		  'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
		  'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
		  'yule']

		See the documentation for scipy.spatial.distance for details on these
		metrics:
		https://docs.scipy.org/doc/scipy/reference/spatial.distance.html.

	p : int, default=2
		Parameter for the Minkowski metric from
		:func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
		is equivalent to using manhattan_distance (l1), and euclidean_distance
		(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

	metric_params : dict, default=None
		Additional keyword arguments for the metric function.

	contamination : 'auto' or float, default='auto'
		The amount of contamination of the data set, i.e. the proportion
		of outliers in the data set. When fitting this is used to define the
		threshold on the scores of the samples.

		- if 'auto', the threshold is determined as in the
		  original paper,
		- if a float, the contamination should be in the range (0, 0.5].

		.. versionchanged:: 0.22
		   The default value of ``contamination`` changed from 0.1
		   to ``'auto'``.

	novelty : bool, default=False
		By default, LocalOutlierFactor is only meant to be used for outlier
		detection (novelty=False). Set novelty to True if you want to use
		LocalOutlierFactor for novelty detection. In this case be aware that
		you should only use predict, decision_function and score_samples
		on new unseen data and not on the training set.

		.. versionadded:: 0.20

	n_jobs : int, default=None
		The number of parallel jobs to run for neighbors search.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.
	
	Attributes
	----------
	labels_ : ndarray of shape (n_samples)
		Anomaly labels for points, 0 for normal points and 1 for anomalies.
		
	scores_ : ndarray of shape (n_samples)
		Anomaly scores for each point. The higher the score, the more likely the
		point is an anomaly.
	
	See Also
	--------
	https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
	"""
	CLASSIFICATIONS = ["voting", "points_score"]
	
	def __init__(self, window: int = 200,
				 stride: int = 1,
				 anomaly_threshold: float = 0.5,
				 classification: str = "auto",
				 n_neighbors: int = 20,
				 algorithm: str = 'auto',
				 leaf_size: int = 30,
				 metric: Union[str, Callable[[list, list], float]] = 'minkowski',
				 p: int = 2,
				 metric_params: dict = None,
				 contamination: Union[str, float] = 'auto',
				 novelty: bool = False,
				 n_jobs: int = None):
		super().__init__(n_neighbors,
						 algorithm=algorithm,
						 leaf_size=leaf_size,
						 metric=metric,
						 p=p,
						 metric_params=metric_params,
						 contamination=contamination,
						 novelty=novelty,
						 n_jobs=n_jobs)
		self.window = window
		self.stride = stride
		self.anomaly_threshold = anomaly_threshold
		self.classification = classification
	
	def fit(self, X, y=None, sample_weight=None) -> None:
		"""Compute the anomalies on the time series.
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.
		sample_weight : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		None
			Fits the model to the data.
		"""
		if self.classification not in self.CLASSIFICATIONS:
			raise ValueError("The classification must be one of",
							 self.CLASSIFICATIONS)
		check_array(X)
		X = np.array(X)
		
		# Project time series onto vector space
		projector = TimeSeriesProjector(self.window, self.stride)
		X_new = projector.fit_transform(X)
		
		# Run vanilla LOF on the vector space of the time series
		super().fit(X_new)
		
		anomalies = np.argwhere(self.negative_outlier_factor_ < self.offset_)
		window_anomalies = np.zeros(X_new.shape[0])
		window_anomalies[anomalies] = 1
		window_scores = - self.negative_outlier_factor_
		
		# TODO: Identical to DBSCAN code, find a way to avoid code duplication
		self.labels_ = np.zeros(X.shape[0])
		self.scores_ = np.zeros(X.shape[0])
		
		# Compute score of each point
		for i in range(window_scores.shape[0]):
			idx = i * self.stride
			self.scores_[idx:idx + self.window] += window_scores[i]
		self.scores_ = self.scores_ / projector.num_windows_
		
		if self.classification == "voting":
			# Anomalies are computed by voting of window anomalies
			for i in range(window_scores.shape[0]):
				if window_anomalies[i] == 1:
					idx = i * self.stride
					self.labels_[idx:idx + self.window] += 1
			self.labels_ = self.labels_ / projector.num_windows_
			
			true_anomalies = np.argwhere(self.labels_ > self.anomaly_threshold)
			self.labels_ = np.zeros(self.labels_.shape)
			self.labels_[true_anomalies] = 1
		else:
			self.labels_[np.argwhere(self.scores_ > self.anomaly_threshold)] = 1
		
		# Min-max normalization
		self.scores_ = self.scores_.reshape((self.scores_.shape[0], 1))
		self.scores_ = MinMaxScaler().fit_transform(self.scores_)
		self.scores_ = self.scores_.reshape(self.scores_.shape[0])
	
	def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
		"""Compute the anomalies on the time series.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The training time series on which we have to train the data.
		y : Ignored
			Not used, present by API consistency by convention.
		sample_weight : Ignored
			Not used, present by API consistency by convention.

		Returns
		-------
		labels
			The labels for the points on the dataset.
		"""
		self.fit(X, y)
		return self.labels_
