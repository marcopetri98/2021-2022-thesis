import numpy as np


def all_indices(list_: list, arg) -> list[int]:
    """Finds all indices of `arg` in `list`, if any.
    
    Parameters
    ----------
    list_ : list
        It is a list in which we want to find occurrences of `arg`.
        
    arg : object
        It is the object we are looking for in `list_`.

    Returns
    -------
    indices : list of int
        It is the list containing all the indices of `list_` containing `arg`.
    """
    indices = [idx
               for idx, elem in enumerate(list_) if elem == arg]
    return indices


def concat_list_array(array: list[np.ndarray]) -> np.ndarray:
    """Concatenates all the ndarray inside the list.
    
    Parameters
    ----------
    array : list[ndarray]
        A list of numpy arrays

    Returns
    -------
    array : ndarray
        The numpy array obtained by concatenation of all the arrays inside the
        list.
    """
    a_final = None
    for a in array:
        if a_final is None:
            a_final = a
        else:
            a_final = np.concatenate((a_final, a))
    return a_final
