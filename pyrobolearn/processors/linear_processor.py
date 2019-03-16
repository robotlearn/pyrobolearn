#!/usr/bin/env python
"""Define the Linear Processor class.
"""

import numpy as np
import torch

from processor import Processor


class LinearProcessor(Processor):
    r"""Linear Processor

    Linear processor is a linear model: :math:`y = ax + b` where :math:`x` is the input, :math:`y` is the output,
    and :math:`a` and :math:`b` are given and fixed coefficients (slope and bias).
    """

    def __init__(self, a, b):
        super(LinearProcessor, self).__init__()
        self.a = torch.Tensor(a)
        self.b = torch.Tensor(b)

    def compute(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            y = self.a * x + self.b
            return y.numpy()
        return self.a * x + self.b
