
import inspect
import types
import numpy as np

# Built-in functions

def hasAttribute(object, name):
    """Check if the given object has an attribute (variable or method) with the given name"""
    return hasattr(object, name)

def hasVariable(object, name):
    """Check if the given object has a variable with the given name"""
    attribute = getattr(object, name, None)
    if attribute is not None:
        if not callable(attribute):
            return True
        # if callable, it might be a callable object, a function, or method
        # A variable can be an object or a function, but not a method.
        return not isinstance(attribute, types.MethodType)  # types.FunctionType
    return False

def hasMethod(object, name):
    """Check if the given object has a method with the given name"""
    method = getattr(object, name, None)
    return inspect.ismethod(method)

def isMethod(object):
    """Check if the given object is a method"""
    return inspect.ismethod(object)

def isClass(object):
    """Check if the given object is a class"""
    return inspect.isclass(object)

def isModule(object):
    """Check if the given object is a module"""
    return inspect.ismodule(object)

def isList(object):
    """Check if the given object is a list"""
    return isinstance(object, list)

def isTuple(object):
    """Check if the given object is a tuple"""
    return isinstance(object, tuple)

def isNumpyArray(object):
    """Check if the given object is a numpy array"""
    return isinstance(object, np.ndarray)

def isDict(object):
    """Check if the given object is a dictionary"""
    return isinstance(object, dict)

def isSet(object):
    """Check the given object is a set"""
    return isinstance(object, set)

def isNone(object):
    """Check if the given object is None"""
    return object is None

def isInt(object):
    """Check if the given object is an integer"""
    return isinstance(object, int)

def isFloat(object):
    """Check if the given object is a float"""
    return isinstance(object, float)

def isStr(object):
    """Check if the given object is a string"""
    return isinstance(object, str)

def isChar(object):
    """Check if the given object is a character"""
    if isinstance(object, str):
        if len(object) == 1:
            return True
    return False

def isBool(object):
    """Check if the given object is a boolean"""
    return isinstance(object, bool)

def isComplex(object):
    """Check if the given object is a complex number"""
    return isinstance(object, complex)