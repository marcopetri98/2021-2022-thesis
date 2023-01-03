import pickle

from mleasy.models import ISavable, BaseModel


class SavableModel(ISavable, BaseModel):
    """Object representing a base model that can be saved."""

    def __init__(self):
        super().__init__()

    def save(self, file_path: str,
             *args,
             **kwargs) -> None:
        params_and_attribs = self._get_all_params()

        if not file_path.endswith(".pickle"):
            file_path += ".pickle"

        with open(file_path, "wb") as file:
            pickle.dump(params_and_attribs, file)

    def load(self, file_path: str,
             *args,
             **kwargs) -> None:
        with open(file_path, "rb") as file:
            params_and_attribs = pickle.load(file)

        self.set_params(**params_and_attribs)
