#!/usr/bin/env python
"""Provide a wrapper around the Scipy optimizers.

References:
    [1] https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

import numpy as np
import scipy

from optimizer import Optimizer

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Scipy(object):
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

    def __init__(self, method='SLSQP'):
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
        self.method = method

    def optimize(self, maxiter=1e6, verbose=True):
        # define objective function to MINIMIZE
        # f = lambda x: -(x.T.dot(C)).dot(x)
        def f(x):
            return -(x.T.dot(C)).dot(x)

        # define initial guess
        x0 = np.ones((M,))  # np.zeros((M,))

        # define 1st constraints: norm of 1
        constraints = [{'type': 'eq', 'fun': lambda x: x.T.dot(x) - 1, 'jac': None, 'args': ()}]

        # define bounds: each vector u have a norm of 1 thus each parameter is between -1 and 1
        bounds = [(-1., 1.)] * M

        # optimize recursively
        evals, evecs = [], []
        messages = {}
        options = {'maxiter': maxiter, 'disp': verbose}
        for i in range(M):
            if i != 0:
                # add orthogonality constraint
                constraints.append({'type': 'eq', 'fun': lambda u: u1.T.dot(u)})

            # minimize --> it returns an instance of OptimizeResult
            result = scipy.optimize.minimize(f, x0, args=(), method=self.method, jac=None, hess=None, bounds=bounds,
                                         constraints=constraints, tol=None, callback=None, options=options)

            print(result.success)
            print(result.message)
            print(result.fun)
            print(result.x)
