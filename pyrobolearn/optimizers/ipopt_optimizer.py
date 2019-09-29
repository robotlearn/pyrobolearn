# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide a wrapper around the non-linear Interior Point optimizer (IPopt).

References:
    [1] https://projects.coin-or.org/Ipopt
"""

import numpy as np

# IPopt optimizer
try:
    import ipopt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install ipopt using the `pyrobolearn/scripts/install_ipopt.sh`."
                                    "If ipopt is already installed, you can install the python wrapper via "
                                    "`pip install ipopt`.")

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NormConstraint(object):

    def __init__(self):
        pass

    def constraint(self, x):
        return x.T.dot(x)

    def jacobian(self, x):
        return 2 * x


class OrthogonalConstraint(object):

    def __init__(self, v):
        self.v = np.copy(v)

    def constraint(self, x):
        return x.T.dot(self.v)

    def jacobian(self, x):
        return self.v


class _IPopt(object):

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.iter_count = 0
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def objective(self, x):
        # objective fct to minimize
        return -x.T.dot(C).dot(x)

    def gradient(self, x):
        # grad of the objective fct
        return -2 * x.T.dot(C)

    def constraints(self, x):
        return np.array([c.constraint(x) for c in self.constraints])

    def jacobian(self, x):
        return np.array([c.jacobian(x) for c in self.constraints])

    # def hessian(self, x):
    #    pass

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                     regularization_size, alpha_du, alpha_pr, ls_trials):
        if self.verbose:
            print("Objective value at iteration #%d: %g" % (iter_count, obj_value))
        self.iter_count = iter_count


class IPopt(Optimizer):
    r"""Interior-Point optimizer

    This is a wrapper around the `ipopt` library. It can be used to solve general nonlinear programming problems of
    the form:

    .. math::

        \min_{x \in R^n} f(x)

    subject to

    .. math::

        g_L \leq g(x) \leq g_U

        x_L \leq  x  \leq x_U

    where :math:`x` are the optimization variables (possibly with upper an lower bounds, :math:`x_U` and :math:`x_L`
    respectively), :math:`f(x)` is the objective function and :math:`g(x)` are the general nonlinear constraints.
    The constraints, :math:`g(x)`, have lower and upper bounds. Note that equality constraints can be specified
    by setting :math:`g^i_L = g^i_U`.

    More info:
    - Check the documentation of the `ipopt.problem` method

    References:
        [1] "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear
            programming", Wachter and Biegler, 2004
        [2] Ipopt: https://projects.coin-or.org/Ipopt
        [3] Ipopt in Python: https://pythonhosted.org/ipopt/
        [4] Repos: https://github.com/coin-or/Ipopt   and   https://pypi.org/project/ipopt/
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the interior point optimizer.
        """
        super(IPopt, self).__init__(*args, **kwargs)

    def optimize(self, parameters, loss, max_iters=1, verbose=False, *args, **kwargs):
        """
        Optimize the given loss function with respect to the given parameters.

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

        # define initial value
        x0 = parameters  # important that the initial value != 0 for the computation of the grad!

        # define (lower and upper) bound constraints
        lb = [-1] * N
        ub = [1] * N

        # define constraints; if upper and lower constraints (resp. cu and cl) are equal then equality constraint
        cl = [1] + [0] * (N - 1)
        cu = [1] + [0] * (N - 1)

        # create ipopt (which contains the objective function, its gradients, and constraints)
        opt = _IPopt(verbose=False)
        opt.add_constraint(NormConstraint())
        opt.add_constraint(OrthogonalConstraint(x))

        # define the nonlinear optimization problem
        self.optimizer = ipopt.problem(n=N, m=len(cl[:i]), problem_obj=opt, lb=lb, ub=ub, cl=cl[:i], cu=cu[:i])

        # solve problem
        x, info = self.optimizer.solve(x0)

        # save the results
        self.best_parameters = x
        self.best_result = info['obj_val']

        return self.best_result, self.best_parameters
