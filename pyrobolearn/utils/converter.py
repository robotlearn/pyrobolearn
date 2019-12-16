#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide converter classes which allows to convert from one certain data type to another.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import quaternion
import collections

__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def roll(lst, shift):
    """Roll elements of a list. This is similar to `np.roll()`"""
    return lst[-shift:] + lst[:-shift]


def numpy_to_torch(tensor):
    """Convert from numpy array to pytorch tensor."""
    return torch.from_numpy(tensor).float()


def torch_to_numpy(tensor):
    """Convert from pytorch tensor to numpy array."""
    if tensor.requires_grad:
        return tensor.detach().numpy()
    return tensor.numpy()


class TypeConverter(object):
    r"""Type Converter class

    It describes how to convert a type to another type, and inversely. For instance, a numpy array to a pytorch Tensor,
    and vice-versa.
    """
    __metaclass__ = ABCMeta

    def __init__(self, from_type, to_type):
        self.from_type = from_type
        self.to_type = to_type

    @property
    def from_type(self):
        return self._from_type

    @from_type.setter
    def from_type(self, from_type):
        if from_type is not None:
            if isinstance(from_type, collections.Iterable):
                for t in from_type:
                    if not isinstance(t, type):
                        raise TypeError("Expecting the from_type to be an instance of 'type'")
            else:
                if not isinstance(from_type, type):
                    raise TypeError("Expecting the from_type to be an instance of 'type'")
        self._from_type = from_type

    @property
    def to_type(self):
        return self._to_type

    @to_type.setter
    def to_type(self, to_type):
        if to_type is not None:
            if isinstance(to_type, collections.Iterable):
                for t in to_type:
                    if not isinstance(t, type):
                        raise TypeError("Expecting the to_type to be an instance of 'type'")
            else:
                if not isinstance(to_type, type):
                    raise TypeError("Expecting the to_type to be an instance of 'type'")
        self._to_type = to_type

    @abstractmethod
    def convert_from(self, data):
        """Convert to the 'from_type'"""
        raise NotImplementedError

    @abstractmethod
    def convert_to(self, data):
        """Convert to the 'to_type'"""
        raise NotImplementedError

    def convert(self, data):
        """
        Convert the data to the other type.
        """
        if isinstance(data, self.from_type): # or self.from_type is None:
            return self.convert_to(data)
        return self.convert_from(data)

    def __call__(self, data):
        """
        Call the convert method, and return the converted data.
        """
        return self.convert(data)


class IdentityConverter(TypeConverter):
    r"""Identity Converter

    Dummy converter which does not convert the data.
    """

    def __init__(self):
        super(IdentityConverter, self).__init__(None, None)

    def convert_from(self, data):
        return data

    def convert_to(self, data):
        return data


class NumpyListConverter(TypeConverter):
    r"""Numpy - list converter

    Convert lists/tuples to numpy arrays, and inversely.
    """

    def __init__(self, convention=0):
        """Initialize the converter.

        Args:
            convention (int): convention to follow if 1D array. 0 to left it untouched, 1 to get column vector (i.e.
                shape=(-1,1)), 2 to get row vector (i.e. shape=(1,-1)).
        """
        super(NumpyListConverter, self).__init__(from_type=(list, tuple), to_type=np.ndarray)

        # check convention
        if not isinstance(convention, int):
            raise TypeError("Expecting an integer for the convention {0,1,2}")
        if convention < 0 or convention > 2:
            raise ValueError("Expecting the convention to belong to {0,1,2}")
        self.convention = convention

    def convert_from(self, data):
        """Convert to list"""
        if isinstance(data, self.from_type):
            return list(data)
        elif isinstance(data, self.to_type):
            if len(data.shape) == 2 and (data.shape[0] == 1 or data.shape[1] == 1):
                return data.ravel().tolist() # flatten data
            return data.tolist()
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to numpy array"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            data = np.array(data)
            if len(data.shape) == 1:
                if self.convention == 0: # left untouched
                    return data
                elif self.convention == 1: # column vector
                    return data[:,np.newaxis]
                else: # row vector
                    return data[np.newaxis,:]
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def reshape(self, data, shape):
        """Reshape the data using the converter. Only valid if data is numpy array."""
        if not isinstance(data, self.to_type):
            data = self.convert_to(data)
        return data.reshape(shape)

    def transpose(self, data):
        """Transpose the data using the converter"""
        if not isinstance(data, self.to_type):
            data = self.convert_to(data)
        return data.T


class QuaternionListConverter(TypeConverter):
    r"""Quaternion - list converter

    Convert a list/tuple to a quaternion, and vice-versa.
    """

    def __init__(self, convention=0):
        """Initialize converter

        Args:
            convention (int): if 0, convert np.quaternion (w,x,y,z) to list [w,x,y,z], and inversely
                              if 1, convert np.quaternion (w,x,y,z) to list [x,y,z,w], and inversely
        """
        super(QuaternionListConverter, self).__init__(from_type=(list, tuple), to_type=np.quaternion)
        if not isinstance(convention, int) or convention < 0 or convention > 1:
            raise TypeError("Expecting convention to be 0 or 1.")
        self.convention = convention

    def convert_from(self, data):
        """Convert to list"""
        if isinstance(data, self.from_type):
            return list(data)
        elif isinstance(data, self.to_type):
            return np.roll(quaternion.as_float_array(data), -self.convention).tolist()
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to quaternion"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            return np.quaternion(*roll(data, -self.convention))
        else:
            raise TypeError("Type not known: {}".format(type(data)))


class QuaternionNumpyConverter(TypeConverter):
    r"""Quaternion - numpy array converter

    Convert a numpy array to a quaternion, and vice-versa.
    """

    def __init__(self, convention=0):
        """Initialize converter

        Args:
            convention (int): if 0, convert np.quaternion (w,x,y,z) to list [w,x,y,z], and inversely
                              if 1, convert np.quaternion (w,x,y,z) to list [x,y,z,w], and inversely
        """
        super(QuaternionNumpyConverter, self).__init__(from_type=np.ndarray, to_type=np.quaternion)
        if not isinstance(convention, int) or convention < 0 or convention > 1:
            raise TypeError("Expecting convention to be 0 or 1.")
        self.convention = convention

    def convert_from(self, data):
        """Convert to numpy array"""
        if isinstance(data, self.from_type):
            return data
        elif isinstance(data, self.to_type):
            return np.roll(quaternion.as_float_array(data), -self.convention)
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to quaternion"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            return np.quaternion(*roll(data.ravel().tolist(), -self.convention))
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def reshape(self, data, shape):
        """Reshape the data using the converter. Only valid if data is numpy array."""
        if not isinstance(data, self.from_type):
            data = self.convert_from(data)
        return data.reshape(shape)

    def transpose(self, data):
        """Transpose the data using the converter"""
        if not isinstance(data, self.from_type):
            data = self.convert_from(data)
        return data.T


class QuaternionPyTorchConverter(TypeConverter):
    r"""Quaternion - pytorch tensor converter

    Convert a pytorch tensor to a quaternion, and vice-versa. Currently, it converts it first to a numpy array and
    then the other type.
    """

    def __init__(self, convention=0):
        """Initialize converter

        Args:
            convention (int): if 0, convert np.quaternion (w,x,y,z) to list [w,x,y,z], and inversely
                              if 1, convert np.quaternion (w,x,y,z) to list [x,y,z,w], and inversely
        """
        super(QuaternionPyTorchConverter, self).__init__(from_type=torch.Tensor, to_type=np.quaternion)
        if not isinstance(convention, int) or convention < 0 or convention > 1:
            raise TypeError("Expecting convention to be 0 or 1.")
        self.convention = convention

    def convert_from(self, data):
        """Convert to pytorch tensor"""
        if isinstance(data, self.from_type):
            return data
        elif isinstance(data, self.to_type):
            return torch.from_numpy(np.roll(quaternion.as_float_array(data), -self.convention))
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to quaternion"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            return np.quaternion(*roll(data.view(-1).data.tolist(), -self.convention))
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def reshape(self, data, shape):
        """Reshape the data using the converter. Only valid if data is numpy array."""
        if not isinstance(data, self.from_type):
            data = self.convert_from(data)
        return data.view(shape)

    def transpose(self, data):
        """Transpose the data using the converter"""
        if not isinstance(data, self.from_type):
            data = self.convert_from(data)
        return data.t()


class NumpyNumberConverter(TypeConverter):
    r"""Numpy - number Converter

    Convert a number to a numpy array of dimension 0 or 1, and vice-versa.
    """

    def __init__(self, dim_array=1):
        super(NumpyNumberConverter, self).__init__(from_type=(int, float), to_type=np.ndarray)

        # dimension array
        if not isinstance(dim_array, int):
            raise TypeError("The 'dim_array' argument should be an integer.")
        if dim_array < 0 or dim_array > 1:
            raise ValueError("The 'dim_array' argument should be 0 or 1.")
        self.dim_array = dim_array

    def convert_from(self, data):
        """Convert to a number"""
        if isinstance(data, self.from_type):
            return data
        elif isinstance(data, self.to_type):
            dim = len(data.shape)
            if dim == 0:
                return data[()]
            elif dim == 1:
                return data[0]
            else:
                raise ValueError("The numpy array should have a shape length of 0 or 1.")
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to numpy array"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            if self.dim_array == 0:
                return np.array(data)
            return np.array([data])
        else:
            raise TypeError("Type not known: {}".format(type(data)))


class PyTorchListConverter(TypeConverter):
    r"""Pytorch - list converter

    Convert lists/tuples to pytorch tensors. Currently, it converts it first to a numpy array and then the other type.
    """

    def __init__(self, convention=0):
        """Initialize the converter.

        Args:
            convention (int): convention to follow if 1D array. 0 to left it untouched, 1 to get column vector (i.e.
                shape=(-1,1)), 2 to get row vector (i.e. shape=(1,-1)).
        """
        super(PyTorchListConverter, self).__init__(from_type=(tuple, list), to_type=torch.Tensor)

        # check convention
        if not isinstance(convention, int):
            raise TypeError("Expecting an integer for the convention {0,1,2}")
        if convention < 0 or convention > 2:
            raise ValueError("Expecting the convention to belong to {0,1,2}")
        self.convention = convention

    def convert_from(self, data):
        """Convert to list"""
        if isinstance(data, self.from_type):
            return list(data)
        elif isinstance(data, self.to_type):
            data = data.numpy() # convert to numpy first
            if len(data.shape) == 2 and (data.shape[0] == 1 or data.shape[1] == 1):
                return data.ravel().tolist() # flatten data
            return data.tolist()
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to pytorch tensor"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            data = np.array(data)
            if len(data.shape) == 1:
                if self.convention == 1: # column vector
                    data = data[:,np.newaxis]
                elif self.convention == 2: # row vector
                    data = data[np.newaxis,:]
            return torch.from_numpy(data)
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def reshape(self, data, shape):
        """Reshape the data using the converter. Only valid if data is numpy array."""
        if not isinstance(data, self.to_type):
            data = self.convert_to(data)
        return data.view(shape)

    def transpose(self, data):
        """Transpose the data using the converter"""
        if not isinstance(data, self.to_type):
            data = self.convert_to(data)
        return data.t()


class PyTorchNumpyConverter(TypeConverter):
    r"""PyTorch - Numpy Converter

    Convert numpy arrays to a pytorch tensors, and vice-versa.
    """

    def __init__(self):
        super(PyTorchNumpyConverter, self).__init__(from_type=np.ndarray, to_type=torch.Tensor)

    def convert_from(self, data):
        """Convert to numpy array"""
        if isinstance(data, self.from_type):
            return data
        elif isinstance(data, self.to_type):
            if data.requires_grad:
                return data.detach().numpy()
            return data.numpy()
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def convert_to(self, data):
        """Convert to pytorch tensor"""
        if isinstance(data, self.to_type):
            return data
        elif isinstance(data, self.from_type):
            return torch.from_numpy(data)
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def reshape(self, data, shape):
        """Reshape the data based on the type using the converter."""
        if isinstance(data, self.from_type): # np
            return data.reshape(shape)
        elif isinstance(data, self.to_type): # torch
            return data.view(shape)
        else:
            raise TypeError("Type not known: {}".format(type(data)))

    def transpose(self, data):
        """Transpose the data using the converter"""
        if isinstance(data, self.from_type): # np
            return data.T
        elif isinstance(data, self.to_type): # torch
            return data.t()
        else:
            raise TypeError("Type not known: {}".format(type(data)))


# class OpenCVNumpyConverter(TypeConverter):
#     pass


if __name__ == '__main__':
    converter = NumpyListConverter()
    print("Using {}".format(converter.__class__.__name__))
    a = np.array(range(4))
    print("on np.array: a={} with type {}".format(a, type(a)))
    b = converter(a)
    print("converter(a) gives: {} with type {}".format(b, type(b)))
    b = converter.convert_from(a)
    print("converter.convert_from(a) gives: {} with type {}".format(b, type(b)))
    b = converter.convert_to(a)
    print("converter.convert_to(a) gives: {} with type {}".format(b, type(b)))

    A = np.array(range(4)).reshape(2, 2)
    print("on numpy matrix: \nA={} with type {}".format(A, type(A)))
    b = converter(A)
    print("converter(a) gives: {} with type {}".format(b, type(b)))
    b = converter.convert_from(A)
    print("converter.convert_from(a) gives: {} with type {}".format(b, type(b)))
    b = converter.convert_to(A)
    print("converter.convert_to(a) gives: \n{} with type {}".format(b, type(b)))
