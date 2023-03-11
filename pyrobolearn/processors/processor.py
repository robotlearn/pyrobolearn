#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Processor class.

Processors are functions that are applied to the inputs (respectively outputs) of an approximator/learning model
before (respectively after) being processed by it. Processors might have parameters but they do not have
trainable/optimizable parameters; the parameters are fixed and given at the beginning.
"""

import numpy as np
import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# define decorator that converts the given numpy array to a torch tensor and return it back to a numpy array if
# specified
def convert_numpy(f):
    def wrapper(self, x, to_numpy=False):
        """Process the given argument.

        Args:
            x (np.array, torch.Tensor): input data.
            to_numpy (bool): If True, it will convert the processed data into a numpy array.
        """

        # convert to torch Tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # call inner function on the given argument
        x = f(self, x)

        # reconvert to numpy array if specified, and return it
        if to_numpy and isinstance(x, torch.Tensor):
            if x.requires_grad:
                return x.detach().numpy()
            return x.numpy()

        # return torch Tensor
        return x

    return wrapper


class Processor(object):
    r"""Processor

    Processors are rules that are applied to the inputs and outputs of a model before being processed by the model
    or after. Processors might have parameters but they do not have trainable/optimizable parameters; the parameters
    are fixed and given at the beginning.
    """

    def __init__(self):
        """Initialize the processor."""
        pass

    def reset(self):
        """Reset the processor."""
        pass

    @convert_numpy
    def compute(self, x):
        """Compute the output given the input :attr:`x`."""
        pass

    def __call__(self, x, to_numpy=False):
        """Alias: call :func:`compute` to compute the output given the input :attr:`x`."""
        return self.compute(x, to_numpy=to_numpy)
