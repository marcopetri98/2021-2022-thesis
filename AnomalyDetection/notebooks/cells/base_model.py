import inspect

class BaseModel(object):
    def __init__(self):
        super().__init__()

    def set_params(self, **params) -> None:
        for key, value in params.items():
            if key not in self.__dict__.keys():
                raise ValueError("Parameter '%s' does not exist in class '%s'. "
                                 "Please, read either the signature or the "
                                 "docs for that class." %
                                 (key, self.__class__.__name__))
            else:
                self.__dict__[key] = value
                
    def get_params(self, deep=True) -> dict:
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
                params_to_return[key] = value
                
        return params_to_return