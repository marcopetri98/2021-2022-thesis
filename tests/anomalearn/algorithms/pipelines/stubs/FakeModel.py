import numpy as np

from anomalearn.algorithms import IParametric, IPredictor, BaseModel, ICopyable
from anomalearn.exceptions import NotTrainedError


class FakeModel(ICopyable, IParametric, IPredictor, BaseModel):
    def __init__(self):
        super().__init__()
        
        self.output_shape = None

    def __repr__(self):
        return "FakeModel()"
        
    def __str__(self):
        return "FakeModel"
    
    def __eq__(self, other):
        return isinstance(other, FakeModel)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self):
        return FakeModel()
        
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self.output_shape = y.shape[1:]
    
    def predict(self, x, *args, **kwargs) -> np.ndarray:
        if self.output_shape is None:
            raise NotTrainedError()
            
        return np.random.rand(x.shape[0], *self.output_shape)
