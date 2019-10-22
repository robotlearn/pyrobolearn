# -*- coding: utf-8 -*-
import inspect
import types
import functools
import re
import numpy as np

# import data structures
from . import data_structures

# import transformations
from . import transformation

# import math utils
from . import math_utils
from . import manifold_utils

# import interpolators
from . import interpolator

# import units
from . import units

# import converters
from . import converter

# import feedback laws
from . import feedback

# import real-time plotting
from . import plotting

# import parsers
from . import parsers

# import parsers
# from . import parsers


# Built-in functions

def has_attribute(object, name):
    """Check if the given object has an attribute (variable or method) with the given name"""
    return hasattr(object, name)


def rsetattr(obj, attr, val):
    """
    Recursively set an attribute. This is useful for nested sub-objects or chained properties.

    Examples:
        class A(object):
            def __init__(self, a=0):
                self.a = a

        class B(object):
            def __init__(self, b):
                self.b = b

        obj = B(b=A())
        rsetattr(obj, 'b.a', 1)  # obj.b.a is set to 1 now

    References:
        - https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Recursively get an attribute. This is useful for nested subobjects or chained properties.

    Examples:
        class A(object):
            def __init__(self, a=0):
                self.a = a

        class B(object):
            def __init__(self, b):
                self.b = b

        obj = B(b=A())
        rgetattr(obj, 'b.a')  # this will access obj.b.a and thus print 0

    References:
        - https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        search = re.search("\[.*\]", attr)  # check for square brackets
        if search is not None:
            begin_idx, last_idx = search.span()
            a = getattr(obj, attr[:begin_idx], *args)
            idx = last_idx
            while idx != begin_idx:
                idx = attr.find(']', begin_idx)
                s = attr[begin_idx+1:idx]
                if re.match("[0-9]+", s):  # check if numeric string
                    if ':' in s:  # check for ':', i.e. slice
                        sl = slice(*[{True: lambda n: None, False: int}[x == ''](x)
                                     for x in (s.split(':') + ['', '', ''])[:3]])
                        a = a[sl]
                    else:  # if no slice
                        a = a[int(s)]
                else:  # string for dictionary
                    a = a[s]
                begin_idx = idx
            return a
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def has_variable(object, name):
    """Check if the given object has a variable with the given name"""
    attribute = getattr(object, name, None)
    if attribute is not None:
        if not callable(attribute):
            return True
        # if callable, it might be a callable object, a function, or method
        # A variable can be an object or a function, but not a method.
        return not isinstance(attribute, types.MethodType)  # types.FunctionType
    return False


def has_method(object, name):
    """Check if the given object has a method with the given name"""
    method = getattr(object, name, None)
    return inspect.ismethod(method)


def is_method(object):
    """Check if the given object is a method"""
    return inspect.ismethod(object)


def is_class(object):
    """Check if the given object is a class"""
    return inspect.isclass(object)


def is_module(object):
    """Check if the given object is a module"""
    return inspect.ismodule(object)


def is_list(object):
    """Check if the given object is a list"""
    return isinstance(object, list)


def is_tuple(object):
    """Check if the given object is a tuple"""
    return isinstance(object, tuple)


def is_numpy_array(object):
    """Check if the given object is a numpy array"""
    return isinstance(object, np.ndarray)


def is_dict(object):
    """Check if the given object is a dictionary"""
    return isinstance(object, dict)


def is_set(object):
    """Check the given object is a set"""
    return isinstance(object, set)


def is_none(object):
    """Check if the given object is None"""
    return object is None


def is_int(object):
    """Check if the given object is an integer"""
    return isinstance(object, int)


def is_float(object):
    """Check if the given object is a float"""
    return isinstance(object, float)


def is_str(object):
    """Check if the given object is a string"""
    return isinstance(object, str)


def is_char(object):
    """Check if the given object is a character"""
    if isinstance(object, str):
        if len(object) == 1:
            return True
    return False


def is_bool(object):
    """Check if the given object is a boolean"""
    return isinstance(object, bool)


def is_complex(object):
    """Check if the given object is a complex number"""
    return isinstance(object, complex)
