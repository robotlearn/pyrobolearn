#!/usr/bin/env python
"""Provide the torch backend API.
"""

import torch
from torch import *
import numpy as np
import tensorflow as tf


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# define decorator that converts the given data structure (numpy array, tensorflow tensor, or torch tensor) to a torch
# tensor.
def to_torch(function):
    """
    Decorator around a given function.

    Args:
        function (callable): function to decorate / wrap.

    Returns:
        callable: function that wraps the initial function by making sure its argument is a torch tensor.
    """

    def wrapper(data, *args, **kwargs):
        """Process the given argument.

        Args:
            data (np.array, torch.Tensor, tf.Tensor): input data.
        """
        if not isinstance(data, (tuple, list)):
            data = [data]

        d = []
        for datum in data:
            # convert to torch Tensor
            if isinstance(datum, torch.Tensor):  # if torch tensor, do nothing
                pass
            elif isinstance(datum, np.ndarray):  # if numpy array, convert to torch tensor
                datum = torch.from_numpy(datum).float()
            elif isinstance(datum, tf.Tensor):   # if tensorflow tensor, convert to torch tensor
                # run the tensorflow session
                sess = tf.Session()
                with sess.as_default():
                    datum = datum.eval()

                # convert the numpy array to torch tensor
                if not isinstance(datum, np.ndarray):
                    raise TypeError("The data returned by evaluating the tensorflow session, is not a numpy array, "
                                    "but instead: {}".format(type(datum)))
                datum = torch.from_numpy(datum).float()
            else:
                raise TypeError("Expecting the input to be a `torch.Tensor`, `np.ndarray`, or `tf.Tensor`, instead "
                                "got: {}".format(type(datum)))
            d.append(datum)

        if len(d) == 1:
            data = d[0]
        else:
            data = d

        # call inner function on the given argument
        data = function(data, *args, **kwargs)

        # return torch Tensor
        return data

    return wrapper


def array(data, dtype=None, copy=True, device=None, requires_grad=False, ndmin=0):
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def inv(data, out=None):
    return torch.inverse(data, out=out)


@to_torch
def concatenate(data, axis=0, out=None):
    return torch.cat(data, dim=axis, out=out)

# np.size vs torch.nelement() vs torch.size()

# TODO
# torch.ger --> torch.outer
# torch.dot only works on 1D vector compared to np.dot, in torch, need to use torch.mm
# missing torch.dstack
# missing torch.vstack
# missing torch.hstack
# np.ndim --> torch.dim()
# np.reshape --> torch.view()


# Tests
if __name__ == '__main__':

    # define function to evaluate tf tensor
    def tf_eval(tensor):
        sess = tf.Session()
        with sess.as_default():
            tensor = tensor.eval()
        return tensor

    # create 3 variables
    a = np.ones(2)
    b = torch.ones(2)
    c = tf.ones(2)

    # concatenate
    numpy_result = np.concatenate((a, a))
    torch_result = torch.cat((b, b))
    tf_result = tf_eval(tf.concat((c, c)))

    print("np.concatenate: {}".format(numpy_result))
    print("torch.cat: {}".format(torch_result))
    print("tf.concat: {}".format(tf_result))

    result_1 = concatenate((a, a))
    result_2 = concatenate((b, b))
    result_3 = concatenate((c, c))
    result_4 = concatenate((a, b, c))

    print("concatenate two np.array: {}".format(result_1))
    print("concatenate two torch.Tensor: {}".format(result_2))
    print("concatenate two tf.Tensor: {}".format(result_3))
    print("concatenate one np.array, one torch.Tensor, one tf.Tensor".format(result_4))
