
# -*- coding: utf-8 -*-
from autograd import *
from autograd.numpy import *
import autograd.numpy as np
import numpy


def array(data, dtype=None, copy=True, device=None, requires_grad=False, ndmin=0):
    return np.array(data, dtype=dtype, copy=copy, ndmin=ndmin)


def inv(data, out=None):
    # TODO: inverse not in autograd.numpy
    if out is None:
        return numpy.linalg.inv(data)

    # inplace
    output = numpy.linalg.inv(data)
    for i in range(len(out)):
        out[i] = output[i]
