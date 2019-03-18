#!/usr/bin/env python
"""Provide a wrapper around the Non-Linear optimizers (NLopt).

References:
    [1] https://nlopt.readthedocs.io/en/latest/
"""

import numpy as np

# NLopt optimizers
try:
    import nlopt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install nlopt via `pip install nlopt`.")

from optimizer import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NLopt(object):
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
        [1] NLopt: https://nlopt.readthedocs.io/en/latest/
        [2] NLopt with Python: with Python: https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/
        [3] Github repo: https://github.com/stevengj/nlopt
    """

    # def __init__(self, model, losses, hyperparameters, seed):
    #     super(NLopt, self).__init__(model, losses, hyperparameters)

    def __init__(self, method, submethod=None, seed=None):

        # define useful variables
        self.results = {1: 'success', 2: 'stop_val reached', 3: 'ftol reached', 4: 'xtol reached',
                        5: 'maxeval reached', 6: 'maxtime reached', -1: 'failure', -2: 'invalid args',
                        -3: 'out of memory', -4: 'roundoff limited', -5: 'forced stop'}

        # define random seed
        nlopt.srand(seed)

        # define which solver to use
        def get_opt(method):
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
        self.opt = get_opt(method)

        # define subsolver to use (if we use the AUGLAG method)
        if method == 'AUGLAG':
            if submethod is None:
                submethod = 'SLSQP'
            elif submethod == 'AUGLAG':
                raise ValueError("Submethod should be different from AUGLAG")
            subopt = get_opt(submethod)
            subopt.set_lower_bounds(-1)
            subopt.set_upper_bounds(1)
            # subopt.set_ftol_rel(1e-2)
            # subopt.set_maxeval(100)
            self.opt.set_local_optimizer(subopt)

    def optimize(self):
        # define objective function and its gradient
        def f(x, grad):
            if grad.size > 0:
                grad[:] = 2 * x.T.dot(C)
            return x.T.dot(C).dot(x)

        # define objective function to maximize
        self.opt.set_max_objective(f)

        # if nlopt.GN_ISRES, we can define the population size
        self.opt.set_population(0)  # by default for ISRES: pop=20*(M+1)

        # define bound constraints (should be between -1 and 1 because the norm should be 1)
        self.opt.set_lower_bounds(-1.)
        self.opt.set_upper_bounds(1.)

        # define norm constraint and its gradient
        def c1(x, grad):
            if grad.size > 0:
                grad[:] = 2 * x
            return (x.T.dot(x) - 1)

        # define orthogonal constraint
        class OrthogonalConstraint(object):

            def __init__(self, v):
                self.v = np.copy(v)

            def constraint(self, x, grad):
                if grad.size > 0:
                    grad[:] = self.v
                return (x.T.dot(self.v))

        # define equality constraints
        self.opt.add_equality_constraint(c1, 0)
        # opt.add_equality_mconstraint(constraints, tol)

        # define stopping criteria
        # self.opt.set_stopval(stopval)
        self.opt.set_ftol_rel(1e-8)
        # opt.set_xtol_rel(1e-4)
        self.opt.set_maxeval(100000)  # nb of iteration
        self.opt.set_maxtime(2)  # time in secs

        # define initial value
        x0 = np.array([0.1] * M)  # important that the initial value != 0 for the computation of the grad!

        evals, evecs, msgs = [], [], {}
        for i in range(M):
            # add constraint
            if i > 0:
                c = OrthogonalConstraint(x)
                self.opt.add_equality_constraint(c.constraint, 0)

            # optimize
            try:
                x = self.opt.optimize(x0)
            except nlopt.RoundoffLimited as e:
                pass

            # save values
            evecs.append(x)  # param vector
            evals.append(self.opt.last_optimum_value())  # max value
            msgs[i] = nlopt_results[self.opt.last_optimize_result()]


