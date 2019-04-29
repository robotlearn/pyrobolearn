#!/usr/bin/env python
"""Define the various decorators used in this framework.
"""

import numpy
import torch

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def convert_numpy(f):
    """Decorator that converts the given numpy array to a torch tensor and return it back to a numpy array if
    specified."""
    def wrapper(self, x, to_numpy=False):
        # convert to torch Tensor if numpy array
        if not isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # call inner function on the given argument
        x = f(self, x)

        # reconvert to numpy array if specified, and return it
        if to_numpy:
            return x.numpy()

        # return torch Tensor
        return x

    return wrapper
