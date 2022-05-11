from typing import Union

import numpy as np
from sklearn.utils import check_array


def check_array_1d(X, array_name: str = None) -> None:
    check_array(X, ensure_2d=False)

    array_name = array_name if array_name is not None else "X"
    X = np.array(X)

    if X.ndim > 1:
        raise ValueError(array_name + " must be 1 dimensional array")


def check_x_y_smaller_1d(X, y, x_name: str = None, y_name: str = None):
    check_array_1d(X, array_name=x_name)
    check_array_1d(y, array_name=y_name)

    x_name = x_name if x_name is not None else "X"
    y_name = y_name if y_name is not None else "y"
    X = np.array(X)
    y = np.array(y)

    if y.size < X.size:
        raise ValueError(x_name + " cannot have more elements than " + y_name)

def check_attributes_exists(estimator,
                            attributes: Union[str, list[str]]) -> None:
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
    for key, value in attributes.items():
        check_attributes_exists(estimator, key)
        attr_val = getattr(estimator, key)
        if value is None:
            if attr_val is None:
                raise RuntimeError("Train the model before calling this method")
        else:
            if attr_val == value:
                raise RuntimeError("Train the model before calling this method")