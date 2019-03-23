#!/usr/bin/env python
"""Define the Linear Processor class.
"""

import torch

from pyrobolearn.processors.processor import Processor, convert_numpy

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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
        super(LinearProcessor, self).__init__()
        self.a = torch.tensor(a, dtype=torch.float)
        self.b = torch.tensor(b, dtype=torch.float)

    def reset(self):
        pass

    @convert_numpy
    def compute(self, x):
        return self.a * x + self.b
