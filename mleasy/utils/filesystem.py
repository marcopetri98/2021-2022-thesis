from pathlib import Path


def find_or_create_dir(path: str) -> None:
    """Find or create the dir at the specified path.
    
    Parameters
    ----------
    path : str
        The path of the dir to search and eventually create.

    Returns
    -------
    None
    """
    path_obj = Path(path)

    if path_obj.name.find(".") != -1:
        raise ValueError("path must point to a directory")
    elif not path_obj.is_dir() and path_obj.exists():
        raise ValueError("path must point to an existing directory or to a "
                         "non existing directory")

    # if the directory does not exist, create it
    if not path_obj.exists():
        path_obj.mkdir(parents=True)
