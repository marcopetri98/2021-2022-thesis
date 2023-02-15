import abc


class EqualityABC(abc.ABC):
    """Abstract class for objects implementing == and !=.
    """
    @abc.abstractmethod
    def __eq__(self, other):
        pass
    
    def __ne__(self, other):
        return self.__eq__(other)
