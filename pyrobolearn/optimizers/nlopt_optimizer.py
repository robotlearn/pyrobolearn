#!/usr/bin/env python
"""Provide a wrapper around the Non-Linear optimizers (NLopt).

References:
    [1] https://nlopt.readthedocs.io/en/latest/
"""

from autograd import numpy as np
from autograd import grad
import torch

# NLopt optimizers
try:
    import nlopt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install nlopt via `pip install nlopt`.")

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NLopt(Optimizer):
    r"""Non-Linear Optimizer

    Non-linear optimizers based on the `nlopt` libraries.

    Here is a brief of lists of the current algorithms implemented:
    *

    Nonlinear optimization algos that can handle nonlinear inequality and EQUALITY constraints are:
    - ISRES (Improved Stochastic Ranking Evolution Strategy) --> global derivative-free
    - COBYLA (Constrained Optimization BY Linear Approximations) --> local derivative-free
    - SLSQP (Sequential Least-SQuares Programming) --> local gradient-based
    - AUGLAG (AUGmented LAGrangian) --> global/local derivative-free/gradient based (determined based on the
        subsidiary algo)

    More information about:
    - algorithms: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

    References:
        - [1] NLopt: https://nlopt.readthedocs.io/en/latest/
        - [2] NLopt with Python: with Python: https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/
        - [3] Github repo: https://github.com/stevengj/nlopt
    """

    def __init__(self, method, submethod=None, seed=None, *args, **kwargs):
        """
        Initialize the non-linear optimizer.

        Args:
            method (str): primary optimization method to be used.
            submethod (str): sub-optimization method to be used in the primary optimization method.
            seed (None, int): random seed
            *args:
            **kwargs:
        """
        super(NLopt, self).__init__(*args, **kwargs)

        # define useful variables
        self.results = {1: 'success', 2: 'stop_val reached', 3: 'ftol reached', 4: 'xtol reached',
                        5: 'maxeval reached', 6: 'maxtime reached', -1: 'failure', -2: 'invalid args',
                        -3: 'out of memory', -4: 'roundoff limited', -5: 'forced stop'}

        # define random seed
        nlopt.srand(seed)

        # define which solver to use
        def get_optimizer(method):
            """
            Get the optimizer associated with the given method.

            Args:
                method (str): optimizer string

            Returns:

            """
            if method == 'ISRES':
                return nlopt.opt(nlopt.GN_ISRES, M)
            elif method == 'COBYLA':
                return nlopt.opt(nlopt.LN_COBYLA, M)
            elif method == 'SLSQP':
                return nlopt.opt(nlopt.LD_SLSQP, M)
            elif method == 'AUGLAG':
                return nlopt.opt(nlopt.AUGLAG, M)
            else:
                raise NotImplementedError("The given method has not been implemented")

        if method is None:
            method = 'SLSQP'
        self.optimizer = get_optimizer(method)

        # define subsolver to use (if we use the AUGLAG method)
        if method == 'AUGLAG':
            if submethod is None:
                submethod = 'SLSQP'
            elif submethod == 'AUGLAG':
                raise ValueError("Submethod should be different from AUGLAG")
            subopt = get_optimizer(submethod)
            subopt.set_lower_bounds(-1)
            subopt.set_upper_bounds(1)
            # subopt.set_ftol_rel(1e-2)
            # subopt.set_maxeval(100)
            self.optimizer.set_local_optimizer(subopt)

    def optimize(self, parameters, loss, max_iters=1, verbose=False, *args, **kwargs):
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
        # define objective function and its gradient
        def f(x, grad):
            loss_value = loss(x)
            if grad.size > 0:
                grad[:] = grad(loss, x)
            return loss_value

        # define objective function to maximize
        self.optimizer.set_min_objective(f)

        # if nlopt.GN_ISRES, we can define the population size
        self.optimizer.set_population(0)  # by default for ISRES: pop=20*(M+1)

        # define bound constraints (should be between -1 and 1 because the norm should be 1)
        self.optimizer.set_lower_bounds(-1.)
        self.optimizer.set_upper_bounds(1.)

        # define norm constraint and its gradient
        def c1(x, grad):
            if grad.size > 0:
                grad[:] = 2 * x
            return x.T.dot(x) - 1

        # define orthogonal constraint
        class OrthogonalConstraint(object):

            def __init__(self, v):
                self.v = np.copy(v)

            def constraint(self, x, grad):
                if grad.size > 0:
                    grad[:] = self.v
                return x.T.dot(self.v)

        # define equality constraints
        self.optimizer.add_equality_constraint(c1, 0)
        # opt.add_equality_mconstraint(constraints, tol)

        # define stopping criteria
        # self.optimizer.set_stopval(stopval)
        self.optimizer.set_ftol_rel(1e-8)
        # opt.set_xtol_rel(1e-4)
        self.optimizer.set_maxeval(100000)  # nb of iteration
        self.optimizer.set_maxtime(2)  # time in secs

        # define initial value
        x0 = np.array([0.1] * M)  # important that the initial value != 0 for the computation of the grad!

        evals, evecs, msgs = [], [], {}
        for i in range(M):
            # add constraint
            if i > 0:
                c = OrthogonalConstraint(x)
                self.optimizer.add_equality_constraint(c.constraint, 0)

            # optimize
            try:
                x = self.optimizer.optimize(x0)
            except nlopt.RoundoffLimited as e:
                pass

            # save values
            evecs.append(x)  # param vector
            evals.append(self.optimizer.last_optimum_value())  # max value
            msgs[i] = nlopt_results[self.optimizer.last_optimize_result()]


