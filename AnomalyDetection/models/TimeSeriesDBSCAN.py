import numpy as np

from models.Clusterer import Clusterer


class TimeSeriesDBSCAN(Clusterer):
	"""Abstract class used to define an unsupervised learner"""
	
	def __init__(self, eps: float = 0.5,
				 min_points: int = 5,
				 metric: str = "euclidean",
				 metric_params: dict = None,
				 algorithm: str = "auto",
				 leaf_size: int = 30,
				 p: float = None,
				 n_jobs: int = None):
		super().__init__()
		self.eps = eps
		self.min_points = min_points
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs
	
	def fit(self, data: np.ndarray,
			*args,
			**kwargs) -> None:
		pass
	
	def get_clusters(self, *args,
					 **kwargs) -> None:
		pass
	
	def get_clustering(self, *args,
					   **kwargs) -> None:
		pass
	
	def set_params(self, eps: float = 0.5,
				   min_points: int = 5,
				   metric: str = "euclidean",
				   metric_params: dict = None,
				   algorithm: str = "auto",
				   leaf_size: int = 30,
				   p: float = None,
				   n_jobs: int = None,
				   *args,
				   **kwargs) -> None:
		self.eps = eps
		self.min_points = min_points
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs
