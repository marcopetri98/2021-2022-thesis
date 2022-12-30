import json
from os.path import exists


def save_py_json(obj_to_save,
                 path: str) -> None:
    """Save a python object to file using json.
    
    Parameters
    ----------
    obj_to_save : object
        Python object to save on a json file.
    
    path : str
        Path where to store the object.

    Returns
    -------
    None
    """
    json_string = json.JSONEncoder().encode(obj_to_save)
    
    with open(path, mode="w") as file_:
        json.dump(json_string, file_)


def load_py_json(path: str) -> object | None:
    """Load a python object from file saved with json.
    
    Parameters
    ----------
    path : str
        Path where the object is stored.

    Returns
    -------
    result : object or None
        The object that has been loaded from file or None in case the path does
        not exist.
    """
    if exists(path):
        with open(path) as file_:
            json_string = json.load(file_)
        
        return json.JSONDecoder().decode(json_string)
    else:
        return None
