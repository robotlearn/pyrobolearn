#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Linear Processor class.
"""

import torch

from pyrobolearn.processors.processor import Processor, convert_numpy

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinearProcessor(Processor):
    r"""Linear Processor

    Linear processor is a linear model: :math:`y = ax + b` where :math:`x` is the input, :math:`y` is the output,
    and :math:`a` and :math:`b` are given and fixed coefficients (slope and bias).
    """

    def __init__(self, a, b):
        """
        Initialize the linear processor.

        Args:
            a (torch.Tensor, np.array): weight
            b (torch.Tensor, np.array): bias
        """
        super(LinearProcessor, self).__init__()
        self.a = torch.tensor(a, dtype=torch.float)
        self.b = torch.tensor(b, dtype=torch.float)

    def reset(self):
        """Reset the linear processor."""
        pass

    @convert_numpy
    def compute(self, x):
        """Compute the linear output given the input :attr:`x`."""
        return self.a * x + self.b
