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
            if key not in self.__dict__:
                raise ValueError("Parameter '%s' does not exist in class '%s'. "
                                 "Please, read either the signature or the "
                                 "docs for that class." %
                                 (key, self.__class__.__name__))
            else:
                self.__dict__[key] = value

    def get_params(self, deep=True) -> dict:
        """Gets all the parameters (public attributes) of the model.

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
        all_parameters = self._get_all_params(deep=deep)
        public_parameters = {key: value
                             for key, value in all_parameters.items()
                             if not key.startswith("_")}
        return public_parameters

    def _get_all_params(self, deep=True) -> dict:
        """Gets all the parameters and attributes of the model.

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
        parameters = vars(self)
        params_to_return = dict()

        for key, value in parameters.items():
            if deep and isinstance(value, BaseModel):
                params_to_return[key] = value.get_params(deep)
            else:
                params_to_return[key] = value

        return params_to_return
