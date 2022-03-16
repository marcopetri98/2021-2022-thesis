"""Package containing base classes used by all or the majority of classes.

One of the most important classes and functions are contained in the Base module.
It contains the definition of the base object from which every object of the
project are derived and the function allowing to extend multiple objects without
breaking the ERROR_KEY order.

Modules
-------
Base
	Contains BaseObject class and mix_keys function used to define project
	objects and to handle multiple inheritance.
"""