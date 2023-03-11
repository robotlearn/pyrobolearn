#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide a wrapper around the Scipy optimizers.

References:
    [1] https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

import numpy as np
import scipy
import scipy.optimize

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Scipy(Optimizer):
    r"""Scipy optimizer

    This uses the `scipy.optimize.minimize` to optimize a given objective function under various bounds and
    constraints. Specifically, it consists of the minimization of a scalar function of one or more variables.
    In general, the optimization problems are of the form:

    .. math::

        \min_{x \in R^n} f(x)

    subject to

    .. math::

        g_i(x) \geq 0, \quad i = 1,...,m
        h_j(x)  =   0, \quad j = 1,...,p

    where :math:`x` is a vector of one or more variables, :math:`g_i(x)` are the inequality constraints, and
    :math:`h_j(x)` are the equality constrains.

    Optionally, the lower and upper bounds for each element in :math:`x` can also be specified using the `bounds`
    argument.

    Several methods/optimizers are available:
    -

    Note that only 'COBYLA' and 'SLSQP' support constraints, where the former only supports inequality constraints.

    References:
        [1] scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def __init__(self, method='SLSQP', *args, **kwargs):
        """
        Initialize the scipy method

        Args:
            method (str, callable):
                - 'Nelder-Mead' :ref:`(see here) <scipy.optimize.minimize-neldermead>`
                - 'Powell'      :ref:`(see here) <scipy.optimize.minimize-powell>`
                - 'CG'          :ref:`(see here) <scipy.optimize.minimize-cg>`
                - 'BFGS'        :ref:`(see here) <scipy.optimize.minimize-bfgs>`
                - 'Newton-CG'   :ref:`(see here) <scipy.optimize.minimize-newtoncg>`
                - 'L-BFGS-B'    :ref:`(see here) <scipy.optimize.minimize-lbfgsb>`
                - 'TNC'         :ref:`(see here) <scipy.optimize.minimize-tnc>`
                - 'COBYLA'      :ref:`(see here) <scipy.optimize.minimize-cobyla>`
                - 'SLSQP'       :ref:`(see here) <scipy.optimize.minimize-slsqp>`
                - 'dogleg'      :ref:`(see here) <scipy.optimize.minimize-dogleg>`
                - 'trust-ncg'   :ref:`(see here) <scipy.optimize.minimize-trustncg>`
                - custom - a callable object (added in version 0.14.0),

        """
        # define optimization method
        # By default, it will be 'BFGS', 'L-BFGS-B', or 'SLSQP' depending on the constraints and bounds
        # If constraints, it can only be 'COBYLA' or 'SLSQP'. COBYLA only supports inequality constraints.
        super(Scipy, self).__init__(*args, **kwargs)
        self.method = method

    def optimize(self, parameters, loss, max_iters=1e6, verbose=False, *args, **kwargs):
        """
        Optimize the given objective function using the optimizer.

        Args:
            parameters (np.array): parameters to optimize.
            loss (callable): callable objective / loss function to minimize.
            bounds (tuple, list, np.array): parameter bounds. E.g. bounds=[0, np.inf]
            max_iters (int): number of maximum iterations.
            verbose (bool): if True, it will display information during the optimization process.
            *args: list of arguments to give to the loss function if callable.
            **kwargs: dictionary of arguments to give to the loss function if callable.

        Returns:
            float, torch.Tensor, np.array: loss scalar value.
            object: best parameters
        """
        N = len(parameters)

        # define initial guess
        x0 = np.ones((N,))  # np.zeros((N,))

        # define 1st constraints: norm of 1
        constraints = [{'type': 'eq', 'fun': lambda x: x.T.dot(x) - 1, 'jac': None, 'args': ()}]

        # define bounds: each vector u have a norm of 1 thus each parameter is between -1 and 1
        bounds = [(-1., 1.)] * N

        # optimize recursively
        evals, evecs = [], []
        messages = {}
        options = {'maxiter': max_iters, 'disp': verbose}
        for i in range(N):
            if i != 0:
                # add orthogonality constraint
                constraints.append({'type': 'eq', 'fun': lambda u: u1.T.dot(u)})

            # minimize --> it returns an instance of OptimizeResult
            result = scipy.optimize.minimize(loss, x0, args=(), method=self.method, jac=None, hess=None, bounds=bounds,
                                             constraints=constraints, tol=None, callback=None, options=options)

            if verbose:
                print(result.success)
                print(result.message)
                print(result.fun)
                print(result.x)

        return
