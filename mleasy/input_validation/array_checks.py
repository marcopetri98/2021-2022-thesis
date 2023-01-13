from typing import Tuple

import numpy as np
from sklearn.utils import check_array


def check_array_general(X,
                        dimensions: int,
                        minimum_samples: Tuple = None,
                        force_all_finite: bool | str = True,
                        array_name: str = None) -> None:
    """Checks that the array satisfies the properties.
    
    Parameters
    ----------
    X : array-like
        The array to be controlled.
        
    dimensions : int
        The number of dimensions that the array must have.
    
    minimum_samples : Tuple, default=None
        The minimum number of elements for each dimension. If None, no minimum
        number of samples is checked.
    
    force_all_finite : bool or {"allow-nan"}, default=True
        If True all elements are forces to be finite values, infinity values and
        NaN values will imply an invalid array. With "allow-nan" the array can
        have finite values and NaN, but not infinity. With False the array can
        have any value.
    
    array_name : str, default=None
        The name of the array to use in exceptions.

    Returns
    -------
    None
    """
    array_name = array_name if array_name is not None else "X"
    
    if minimum_samples is not None and len(minimum_samples) != dimensions:
        raise ValueError("minimum_samples must have at least shape elements, or"
                         " it can be None")
    
    if dimensions == 2:
        if minimum_samples is not None:
            check_array(X,
                        ensure_min_samples=minimum_samples[0],
                        ensure_min_features=minimum_samples[1],
                        force_all_finite=force_all_finite)
        else:
            check_array(X, force_all_finite=force_all_finite)
    else:
        check_array(X, ensure_2d=False, force_all_finite=force_all_finite)
        
        np_arr = np.array(X)
        
        if len(np_arr.shape) != dimensions:
            raise ValueError(array_name + " doesn't have the specified shape")
        
        if minimum_samples is not None:
            for i in range(len(np_arr.shape)):
                if np_arr.shape[i] < minimum_samples[i]:
                    raise ValueError(array_name + " have too few elements "
                                                  " at dimension " + str(i))


def check_array_1d(X, array_name: str = None, force_all_finite: bool | str = True) -> None:
    check_array(X, ensure_2d=False, force_all_finite=force_all_finite)
    
    array_name = array_name if array_name is not None else "X"
    X = np.array(X)
    
    if X.ndim > 1:
        raise ValueError(array_name + " must be 1 dimensional array")


def check_x_y_smaller_1d(X, y, x_name: str = None, y_name: str = None, force_all_finite: bool | str = True):
    check_array_1d(X, array_name=x_name, force_all_finite=force_all_finite)
    check_array_1d(y, array_name=y_name, force_all_finite=force_all_finite)
    
    x_name = x_name if x_name is not None else "X"
    y_name = y_name if y_name is not None else "y"
    X = np.array(X)
    y = np.array(y)
    
    if y.size < X.size:
        raise ValueError(x_name + " cannot have more elements than " + y_name)
