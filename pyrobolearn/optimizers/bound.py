#!/usr/bin/env python
r"""Define the bounds in an optimization problem.

A constrained optimization problem is generally given by:

.. math:: \min_{x \in R^n} f(x)

subject to

.. math::

        g_L \leq g(x) \leq g_U
        x_L \leq  x  \leq x_U

where :math:`f` is the objective function, :math:`x` are the variables that are being optimized, :math:`(x_L, x_U)`
are the lower and upper bound on these variables, :math:`g` is a constraint function that maps the variables :math:`x`
to another space, and :math:`(g_L, g_U)` are the lower and upper bound in that space.

References:
    [1] https://nlopt.readthedocs.io/en/latest/
"""

import torch
import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Bound(object):
    r"""Lower and upper bounds

    """

    def __init__(self, lower, upper):
        """
        Initialize the bounds.

        Args:
            lower (np.array, torch.Tensor, float, int): lower bound
            upper (np.array, torch.Tensor, float, int): upper bound
        """
        self._lower = lower
        self._upper = upper

    def __call__(self, variables):
        return self._lower <= variables <= self._upper
