import abc
import logging


__module_logger = logging.getLogger(__name__)


def _check_methods(klass, *methods):
    """Check if the class or any of its parents has all specified methods.
    
    Between the parents that are searched, object is not considered. We consider
    that the class has a method only if at least one class more specific than
    object has the method. That is: it does not use the default implementation.
    
    Parameters
    ----------
    klass : class
        The class to be checked.
    
    methods : list[str]
        The methods that the class must have.

    Returns
    -------
    are_there_methods
        NotImplemented if there is no method or a method is None. True if all
        methods are present.
    """
    __module_logger.debug(f"received class {klass}")
    __module_logger.debug(f"class.__dict__ = {klass.__dict__}")
    # exclude object an maintain order
    mro = [e for e in klass.__mro__ if e is not object]
    for method in methods:
        for base in mro:
            if method in base.__dict__:
                if base.__dict__[method] is None:
                    return NotImplemented
                else:
                    break
        else:
            return NotImplemented
    return True


class EqualityABC(abc.ABC):
    """Abstract class for objects implementing == and !=.
    """
    @abc.abstractmethod
    def __eq__(self, other):
        pass
    
    def __ne__(self, other):
        return self.__eq__(other)

    @classmethod
    def __subclasshook__(cls, other):
        if cls is EqualityABC:
            return _check_methods(other, "__eq__", "__ne__")
        return NotImplemented
