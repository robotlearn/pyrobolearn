#!/usr/bin/env python
r"""Define the optimization problem.

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

from pyrobolearn.optimizers.objective import Objective
from pyrobolearn.optimizers.constraint import Constraint
from pyrobolearn.optimizers.bound import Bound

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class OptimizationProblem(object):
    r"""Optimization problem

    A constrained optimization problem is generally given by:

    .. math:: \min_{x \in R^n} f(x)

    subject to

    .. math::

            g_L \leq g(x) \leq g_U
            x_L \leq  x  \leq x_U

    where :math:`f` is the objective function, :math:`x` are the variables that are being optimized, :math:`(x_L, x_U)`
    are the lower and upper bound on these variables, :math:`g` is a constraint function that maps the variables
    :math:`x` to another space, and :math:`(g_L, g_U)` are the lower and upper bound in that space.
    """

    def __init__(self, loss, bounds, constraints):
        """
        Initialize the optimization problem.

        Args:
            loss (Objective): objective / loss function.
            bounds ((list of) Bound): bounds
            constraints ((list of) Constaint): constraints
        """
        # check loss function
        if not isinstance(loss, Objective):
            loss = Objective(loss)
        self._loss = loss

        # check bounds
        if not isinstance(bounds, list):
            bounds = [bounds]
        for i, bound in enumerate(bounds):
            if isinstance(bound, tuple) and len(bound) == 2:
                bounds[i] = Bound(lower=bound[0], upper=bound[1])
            elif not isinstance(bound, Bound):
                raise TypeError("Expecting the given bound to be an instance of `Bound`, instead got: "
                                "{}".format(type(bound)))
        self._bounds = bounds

        # check constraints
        if not isinstance(constraints, list):
            constraints = [constraints]
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, tuple) and len(constraint) == 3:
                constraint, lower, upper = constraint
                constraints[i] = Constraint(constraint, lower=lower, upper=upper)
            if not isinstance(constraint, Constraint):
                raise TypeError("Expecting the given constraint to be an instance of `Constraint`, instead got: "
                                "{}".format(type(constraint)))
        self._constraints = constraints

    @property
    def objective(self):
        """Return the objective function."""
        return self._loss

    loss = objective

    @property
    def bounds(self):
        """Return the bounds."""
        return self._bounds

    @property
    def constraints(self):
        """Return the constraints."""
        return self._constraints

    def __call__(self, variables, *args, **kwargs):
        """Return the objective function."""
        return self.loss(variables, *args, **kwargs)
