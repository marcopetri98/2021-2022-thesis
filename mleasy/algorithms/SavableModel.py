from pathlib import Path

from . import ISavable, BaseModel
from ..utils import save_py_json, find_or_create_dir, load_py_json


class SavableModel(ISavable, BaseModel):
    """Object representing a base model that can be saved.
    
    If an object can be saved, there are either parameters, hyperparameters or
    configuration parameters that can be saved. If none of the previous is
    present, a model must not be savable.
    """
    __json_file = "savable_model.json"

    def __init__(self):
        super().__init__()

    def save(self, path: str,
             *args,
             **kwargs) -> None:
        find_or_create_dir(path)
        path_obj = Path(path)
            
        json_objects = self.get_params(deep=False)
        save_py_json(json_objects, str(path_obj / self.__json_file))

    def load(self, path: str,
             *args,
             **kwargs) -> None:
        path_obj = Path(path)

        if not path_obj.joinpath(self.__json_file).is_file():
            raise ValueError("path directory is not valid. It must contain "
                             f"these files: {self.__json_file}")

        json_objects: dict = load_py_json(str(path_obj / self.__json_file))
        self.set_params(**json_objects)
