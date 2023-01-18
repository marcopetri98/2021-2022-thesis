import pickle

from . import ISavable, BaseModel


class SavableModel(ISavable, BaseModel):
    """Object representing a base model that can be saved."""

    def __init__(self):
        super().__init__()

    def save(self, path: str,
             *args,
             **kwargs) -> None:
        params_and_attribs = self._get_all_params()

        if not path.endswith(".pickle"):
            path += ".pickle"

        with open(path, "wb") as file:
            pickle.dump(params_and_attribs, file)

    def load(self, path: str,
             *args,
             **kwargs) -> None:
        with open(path, "rb") as file:
            params_and_attribs = pickle.load(file)

        self.set_params(**params_and_attribs)
