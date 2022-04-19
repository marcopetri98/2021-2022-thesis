# Python imports

# External imports

# Project imports
from typing import Union


def check_attributes_exists(estimator,
							attributes: Union[str, list[str]]) -> None:
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

def check_not_default_attributes(estimator,
								 attributes: dict) -> None:
	"""Checks if the attributes have the default not trained value.
	
	It raises an exception if at least one of the attribute has the default not
	trained value.
	
	Parameters
	----------
	estimator : object
		The estimator on which we want to verify if the attribute is set.
		
	attributes : dict
		Keys are string representing the attributes' names and the values are
		the standard values used when the estimator has not been trained yet.

	Returns
	-------
	None
	"""
	for key, value in attributes.items():
		check_attributes_exists(estimator, key)
		attr_val = getattr(estimator, key)
		if value is None:
			if attr_val is None:
				raise RuntimeError("Train the model before calling this method")
		else:
			if attr_val == value:
				raise RuntimeError("Train the model before calling this method")
