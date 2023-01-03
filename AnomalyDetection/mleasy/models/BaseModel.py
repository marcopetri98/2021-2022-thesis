import inspect


class BaseModel(object):
    """Object representing a general model"""

    def __init__(self):
        super().__init__()

    def set_params(self, **params) -> None:
        """Modify the parameters of the object.

        Parameters
        ----------
        params
            The dictionary of the parameters to modify.

        Returns
        -------
        None
        """
        for key, value in params.items():
            if key not in self.__dict__.keys():
                raise ValueError("Parameter '%s' does not exist in class '%s'. "
                                 "Please, read either the signature or the "
                                 "docs for that class." %
                                 (key, self.__class__.__name__))
            else:
                self.__dict__[key] = value

    def get_params(self, deep=True) -> dict:
        """Gets all the parameters of the model defined in init.

        Parameters
        ----------
        deep : bool, default=True
            States whether the method must return also the parameters of nested
            base models.

        Returns
        -------
        param_dict : dict
            Dictionary with parameters' names as keys and values as values.
        """
        init = getattr(self, "__init__")
        if init is object.__init__:
            return {}

        init_signature = inspect.signature(init)
        parameters = [p
                      for p in init_signature.parameters.values()
                      if p.name != "self" and p.kind == p.POSITIONAL_OR_KEYWORD]
        parameters_name = [name.name
                           for name in parameters]
        params_to_return = {}
        for key, value in self.__dict__.items():
            if key in parameters_name:
                if deep and isinstance(value, BaseModel):
                    params_to_return[key] = value.get_params(deep)
                elif not isinstance(value, BaseModel):
                    params_to_return[key] = value

        return params_to_return
