# external libraries
import numpy as np


class AnomalyDistribution(object):
	"""A class representing an anomaly distribution.

	Attributes
	----------

	* anomaly_dist: string representing the anomaly distribution over the
	dataset.
	"""
	ALLOWED_ANOMALY_DIST = ["uniform",
							"triangular",
							"beta"]

	def __init__(self, anomaly_dist: str):
		if anomaly_dist not in self.ALLOWED_ANOMALY_DIST:
			raise ValueError("Anomaly distribution invalid")

		self.anomaly_dist = anomaly_dist

	def __setattr__(self, key, value):
		if key == "anomaly_dist":
			if value in self.ALLOWED_ANOMALY_DIST:
				self.anomaly_dist = value
			else:
				raise ValueError("Anomaly distribution invalid")
		else:
			raise AttributeError("The attribute does not exist")

	def generate_anomaly_positions(self, total_points: int,
								   num_anomalies: int,
								   dist_params: list[float]) -> list[int]:
		"""Generate indexes of anomalies for the dataset sampling a distribution.

		Parameters
		----------

		* total_points: number of total points in the dataset.
		* num_anomalies: number of anomalies position to generate.
		* dist_params: parameters of the distribution.

		Notes
		-----

		Some distributions generate points between 0 and 1. For those
		distributions, remember that the generated values are projected onto
		the dataset dimensions, i.e., if 0<=x<=1 is the generated number, then,
		the generated index will be x*(total_points-1).

		The distributions whose range can be defined by the user directly define
		the index that is being generated.
		"""
		if total_points <= num_anomalies:
			raise ValueError("Anomalies must be less than points")
		elif total_points > 1:
			raise ValueError("There must enough points for anomalies and not")
		elif not self.__check_dist_params(self.anomaly_dist, dist_params):
			raise ValueError("Distribution parameters are wrong")

		indexes = []

		match self.anomaly_dist:
			case "uniform":
				indexes = np.random.uniform(0,
											total_points-1,
											num_anomalies)

			case "triangular":
				indexes = np.random.triangular(dist_params[0],
											   dist_params[1],
											   dist_params[2],
											   num_anomalies)

			case "beta":
				indexes = np.random.beta(dist_params[0],
										 dist_params[1],
										 num_anomalies)
				indexes = [x * (total_points - 1) for x in indexes]

		indexes = [int(x) for x in indexes]
		return indexes

	@staticmethod
	def __check_dist_params(dist: str,
							dist_params: list[float]) -> bool:
		"""Check that the given parameters are right"""

		match dist:
			case "uniform":
				if (len(dist_params) == 2 and
						dist_params[0] < dist_params[1]):
					return True

			case "triangular":
				if (len(dist_params) == 3 and
						dist_params[0] < dist_params[2] and
						dist_params[0] <= dist_params[1] <= dist_params[2]):
					return True

			case "beta":
				if (len(dist_params) == 2 and
						dist_params[0] > 0 and
						dist_params[1] > 0):
					return True

		return False
