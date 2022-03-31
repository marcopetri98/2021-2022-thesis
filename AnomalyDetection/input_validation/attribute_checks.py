# Python imports

# External imports

# Project imports
from typing import Union


def check_attributes_exists(estimator,
							attributes: Union[str, list[str]]):
	"""Checks is the attributes are defined in estimator.

	Parameters
	----------
	estimator : object
		The estimator on which we want to verify if the attribute is set.

	attributes : str or list of str
		The attributes to check the existence in the estimator.

	Returns
	-------
	None
	"""
	if isinstance(attributes, str):
		if attributes not in estimator.__dict__.keys():
			raise ValueError("%s does not have attribute %s" %
							 (estimator.__class__, attributes))
	else:
		for attribute in attributes:
			if attribute not in estimator.__dict__.keys():
				raise ValueError("%s does not have attribute %s" %
								 (estimator.__class__, attribute))
