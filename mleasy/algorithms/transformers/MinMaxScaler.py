from __future__ import annotations

import pickle
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler as scikitMinMaxScaler

from .. import ITransformer, IParametric, SavableModel, ICopyable
from ...exceptions import InvalidInputShape
from ...utils import find_or_create_dir


class MinMaxScaler(ICopyable, ITransformer, IParametric, SavableModel):
    """Min max scaler wrapper for `scikit-learn`.
    
    Attributes
    ----------
    _min_max_scaler : scikit-learn MinMaxScaler
        It is an instance of the scikit-learn `MinMaxScaler`.
    """
    __scikit_file = "min_max_scaler.pickle"
    
    def __init__(self, feature_range: Tuple[Number, Number] = (0, 1),
                 copy: bool = True,
                 clip: bool = False):
        super().__init__()
        
        self._min_max_scaler = scikitMinMaxScaler(feature_range=feature_range,
                                                  copy=copy,
                                                  clip=clip)
    
    @property
    def feature_range(self):
        return self._min_max_scaler.feature_range
    
    @feature_range.setter
    def feature_range(self, value):
        self._min_max_scaler.feature_range = value
        
    @property
    def copy_attribute(self):
        return self._min_max_scaler.copy
    
    @copy_attribute.setter
    def copy_attribute(self, value):
        self._min_max_scaler.copy = value
        
    @property
    def clip(self):
        return self._min_max_scaler.clip
    
    @clip.setter
    def clip(self, value):
        self._min_max_scaler.clip = value
    
    @property
    def scale_adjustment(self):
        try:
            return self._min_max_scaler.scale_
        except AttributeError:
            return None
    
    @property
    def min_adjustment(self):
        try:
            return self._min_max_scaler.min_
        except AttributeError:
            return None
        
    @property
    def seen_data_min(self):
        try:
            return self._min_max_scaler.data_max_
        except AttributeError:
            return None
        
    @property
    def seen_data_max(self):
        try:
            return self._min_max_scaler.data_min_
        except AttributeError:
            return None
    
    @property
    def seen_data_range(self):
        try:
            return self._min_max_scaler.data_range_
        except AttributeError:
            return None
    
    @property
    def seen_features_in(self):
        try:
            return self._min_max_scaler.n_features_in_
        except AttributeError:
            return None
    
    @property
    def seen_samples_in(self):
        try:
            return self._min_max_scaler.n_samples_seen_
        except AttributeError:
            return None
    
    @property
    def seen_features_names_in(self):
        try:
            return self._min_max_scaler.feature_names_in_
        except AttributeError:
            return None
        
    def __repr__(self):
        return f"MinMaxScaler(feature_range={self.feature_range},copy={self.copy_attribute},clip={self.clip})"
    
    def __str__(self):
        return "MinMaxScaler"
    
    def copy(self) -> MinMaxScaler:
        """Copies the object.
        
        Note that since scikit-learn does not provide standard `save` and `load`
        methods for objects, and it does not provide a complete copy method,
        deepcopy will be used.
        
        Returns
        -------
        new_object : MinMaxScaler
            The copied object.
        """
        new = MinMaxScaler(self.feature_range, self.copy_attribute, self.clip)
        new._min_max_scaler = deepcopy(self._min_max_scaler)
        return new
        
    def save(self, path: str,
             *args,
             **kwargs) -> Any:
        find_or_create_dir(path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "wb") as f:
            pickle.dump(self._min_max_scaler, f)
    
    def load(self, path: str,
             *args,
             **kwargs) -> Any:
        find_or_create_dir(path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "rb") as f:
            self._min_max_scaler = pickle.load(f)
        
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self._min_max_scaler.fit(x)
        
    def transform(self, x, *args, **kwargs) -> np.ndarray:
        if x.shape[1] != self.seen_data_max.shape[0]:
            raise InvalidInputShape(("n_points", self.seen_data_max.shape[0]), x.shape)
        
        return self._min_max_scaler.transform(x)
