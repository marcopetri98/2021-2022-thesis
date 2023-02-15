import abc


class ObjectWithEquality(object):
    def __init__(self):
        super().__init__()
    
    def __eq__(self, other):
        raise NotImplementedError
    
    def __ne__(self, other):
        raise NotImplementedError


class ObjectNoMoreWithEquality(ObjectWithEquality):
    def __init__(self):
        super().__init__()
    
    __eq__ = None


class ObjectWithoutEquality(abc.ABC):
    def __init__(self):
        super().__init__()
    
    def __eq__(self, other):
        raise NotImplementedError


class ObjectWithoutEquality2(ObjectWithoutEquality):
    def __init__(self):
        super().__init__()


class ObjectWithEqualityInherit(ObjectWithoutEquality2):
    def __init__(self):
        super().__init__()
    
    def __ne__(self, other):
        raise NotImplementedError
