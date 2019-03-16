#!/usr/bin/env python
"""Provide various optimizers.

Optimizers allows to optimize (i.e. minimize or maximize) a utility function (also known as objective function,
fitness function, loss, etc.) with or without constraints and bounds. Notably, it assumes the functions have some
parameters that the optimizer can update.

Mathematically, this is described as:

.. math:: \min_{x \in R^n} f(x)

subject to

.. math::

    g_i(x) \geq 0, \quad i = 1,...,m
    h_j(x)  =   0, \quad j = 1,...,p
    x_l \leq x \leq x_u

For trajectory optimization, check "An Introduction to Trajectory Optimization: How to do your own Direct Collocation".
"""

# TODO: trajectory optimization

from abc import ABCMeta, abstractmethod

# Numpy with autograd
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

# Scipy optimizer
import scipy

# Pytorch optimizers
import torch.nn as nn
import torch.optim as optim

# NLopt optimizers
try:
    import nlopt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install nlopt via `pip install nlopt`.")

# IPopt optimizer
try:
    import ipopt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install ipopt using the `pyrobolearn/scripts/install_ipopt.sh`."
                                    "If ipopt is already installed, you can install the python wrapper via "
                                    "`pip install ipopt`.")

# CVXOPT
# import cvxopt
# CVXPY: nice wrapper around cvxopt
# import cvxpy
# Quadprog
# import quadprog

# QPsolvers optimizers: unified Python interface for multiple QP solvers (cvxopt, cvxpy, quadprog,...)
try:
    import qpsolvers
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install qpsolvers directly via 'pip install qpsolvers'.")

# Bayesian optimization
try:
    import GPy
    import GPyOpt
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install GPy/GPyOpt directly via 'pip install GPy' and "
                                    "'pip install GPyOpt'.")

# CMA-ES
try:
    import cma
except ImportError as e:
    raise ImportError(e.__str__() + "\n HINT: you can install CMA-ES or `pycma` directly via 'pip install cma'.")


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


##################################################################
#                           OPTIMIZER                            #
##################################################################


class Optimizer(object):
    r"""Optimizer abstract class

    This is an abstract class from which all optimizers inherit from. Most of the child optimizer classes are wrappers
    around the original optimizer. This is to provide the same common interface to all the optimizers, and convert
    seamlessly to the correct data types.

    Optimizers are often given as a parameter to the learning algorithms, but can also be used of out of the box
    directly on models. Several original optimizers can be given to the learning algorithm which will automatically
    wrap the optimizer with the corresponding wrapper to provide a common interface.

    In their most natural form, optimizers are used to ... optimization process which can be described mathematically
    by:

    .. math::

        \min_{\theta} J(\theta) \mbox{ subj. to constraints}

        \max_{\theta} J(\theta)

    Several optimizers provides

    The list of optimizers available are from the following libraries:
    * nlopt
    * ipopt
    * torch.optim
    * GPy / GPyOpt
    * scipy.optimize
    * qpsolvers (which includes cvxpy)
    * cmaes
    * pso

    Each one of them expect a certain kind of type of the parameters.

    Optimizers can be divided into:
    * global vs local
    * derivative-free vs gradient-based
    * without constraints vs with (equality and/or inequality) constraints

    Many implemented optimizers that can be found online are specific to a certain type of learning model.
    If necessary, a conversion or wrapping process is carried out to make the optimizer work with the given learning
    model.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, losses, hyperparameters):
        """

        :param model: a certain type of model. If the original model is given instead of an instance of `Model`, then
                      the model will be wrapped appropriately.
        :param losses:
        :param hyperparameters:
        """
        pass


##################################################################
#                             SCIPY                              #
##################################################################

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


##################################################################
#                    Quadratic Programming                       #
##################################################################

# class CVXOPT(Optimizer):
#     r"""Convex Optimizer
#
#     Note: cvxpy module is a nice wrapper around cvxopt that follows paradigm of a disciplined convex programming.
#
#     References:
#         [1] Python Software for Convex Optimization: https://cvxopt.org/
#         [2] Github repo: https://github.com/cvxopt/cvxopt
#     """
#     pass
#
#
# class CVXPY(Optimizer):
#     r"""Convex Optimizer
#
#     References:
#         [1] CVXPY: http://www.cvxpy.org/
#         [2] Github repo: https://github.com/cvxgrp/cvxpy
#     """
#     pass
#
#
# class QuadProg(object):
#     r"""Quadprog
#
#     References:
#         [1] Github repo: https://github.com/rmcgibbo/quadprog
#     """
#     pass

class QP(object):
    r"""Quadratic Programming solvers

    This class uses the `qpsolvers` which is a unified Python interface for multiple QP solvers [1,2].

    .. math::

        \min_{x \in R^n} \frac{1}{2} x^T P x + q^T x

    subject to

    .. math::

        Gx \leq h
        Ax = b

    where :math:`x` is the vector of optimization variables, the matrix :math:`P` and vector :math:`q` are used to
    define any quadratic objective function on these variables, while the matrix-vector couples :math:`(G,h)` and
    :math:`(A,b)` respectively define inequality and equality constraints. Vector inequalities apply coordinate by
    coordinate [1].

    - Dense solvers:
        - CVXOPT
        - CVXPY
        - qpOASES
        - quadprog
    - Sparse solvers:
        - ECOS as wrapped by CVXPY
        - Gurobi
        - MOSEK
        - OSQP

    Check the available solvers by calling `print(qpsolvers.available_solvers)`.

    Notes: Many solvers (including CVXOPT, OSQP and quadprog) assume that `P` is a symmetric matrix, and may return
    erroneous results when that is not the case. You can set ``sym_proj=True`` to project `P` on its symmetric part,
    at the cost of some computation time.

    References:
        [1] QP in Python: https://scaron.info/blog/quadratic-programming-in-python.html
        [2] Github repo: https://github.com/stephane-caron/qpsolvers
    """

    def __init__(self, method='quadprog'):
        """
        Initialize the QP solver.

        Args:
            method (str): ['cvxopt', 'cvxpy', 'ecos', 'gurobi', 'mosek', 'osqp', 'qpoases', 'quadprog']
        """
        solvers = set(qpsolvers.available_solvers)
        if len(solvers) == 0:
            raise ValueError("No QP solvers have been found on this computer. Please install one of the QP modules")
        if method not in solvers:
            method = 'quadprog'
        self.method = method

        # check methods that require a symmetric matrix for P
        methods = ['cvxopt', 'osqp', 'quadprog']
        self.sym_proj = True if self.method in set(methods) else False

    def is_symmetric(self, X, tol=1e-8):
        return np.allclose(X, X.T, atol=tol)

    def optimize(self, P, q, x0=None, G=None, h=None, A=None, b=None):
        return qpsolvers.solve_qp(P, q, G, h, A, b, solver=self.method, initvals=x0, sym_proj=self.sym_proj)


##################################################################
#                            NLOPT                               #
##################################################################

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
        opt.add_equality_constraint(c1, 0)
        # opt.add_equality_mconstraint(constraints, tol)

        # define stopping criteria
        # opt.set_stopval(stopval)
        opt.set_ftol_rel(1e-8)
        # opt.set_xtol_rel(1e-4)
        opt.set_maxeval(100000)  # nb of iteration
        opt.set_maxtime(2)  # time in secs

        # define initial value
        x0 = np.array([0.1] * M)  # important that the initial value != 0 for the computation of the grad!

        evals, evecs, msgs = [], [], {}
        for i in range(M):
            # add constraint
            if i > 0:
                c = OrthogonalConstraint(x)
                opt.add_equality_constraint(c.constraint, 0)

            # optimize
            try:
                x = opt.optimize(x0)
            except nlopt.RoundoffLimited as e:
                pass

            # save values
            evecs.append(x)  # param vector
            evals.append(opt.last_optimum_value())  # max value
            msgs[i] = nlopt_results[opt.last_optimize_result()]


##################################################################
#                            IPOPT                               #
##################################################################

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

    def __init__(self, model, losses, hyperparameters):
        super(IPopt, self).__init__(model, losses, hyperparameters)

    def optimize(self):
        # define initial value
        x0 = np.array([0.1] * N)  # important that the initial value != 0 for the computation of the grad!

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
        nlp = ipopt.problem(n=N, m=len(cl[:i]), problem_obj=opt, lb=lb, ub=ub, cl=cl[:i], cu=cu[:i])

        # solve problem
        x, info = nlp.solve(x0)

        return x


##################################################################
#                      PYTORCH OPTIMIZERS                        #
##################################################################


class PyTorchOpt(Optimizer):
    r"""PyTorch Optimizers

    This is a wrapper around the optimizers from pytorch.
    """

    def __init__(self, model, losses, hyperparameters):
        super(PyTorchOpt, self).__init__(model, losses, hyperparameters)

    def add_constraint(self):
        # it will add a constraint as the augmented lagrangian
        pass


class Adam(object):
    r"""Adam Optimizer

    References:
        [1] "Adam: A Method for Stochastic Optimization", Kingma et al., 2014
    """

    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.max_grad_norm = max_grad_norm

    def reset(self):
        self.optimizer = None

    def optimize(self, params, loss):
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=self.betas, eps=self.eps,
                                        weight_decay=self.weight_decay, amsgrad=self.amsgrad)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class Adadelta(object):
    r"""Adadelta Optimizer

    References:
        [1] "ADADELTA: An Adaptive Learning Rate Method", Zeiler, 2012
    """

    def __init__(self, learning_rate=1., rho=0.9, eps=1e-6, weight_decay=0, max_grad_norm=None): #0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.Adadelta(params, lr=self.learning_rate, rho=self.rho, eps=self.eps,
                                            weight_decay=self.weight_decay)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class Adagrad(object):
    r"""Adagrad Optimizer

    References:
        [1] "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", Duchi et al., 2011
    """

    def __init__(self, learning_rate=0.01, learning_rate_decay=0, weight_decay=0, initial_accumumaltor_value=0,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumumaltor_value
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.Adagrad(params, lr=self.learning_rate, lr_decay=self.learning_rate_decay,
                                           weight_decay=self.weight_decay,
                                           initial_accumulator_value=self.initial_accumulator_value)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class RMSprop(object):
    r"""RMSprop

    References:
        [1] "RMSprop:  Divide the gradient by a running average of its recent magnitude" (lecture 6.5), Tieleman and
            Hinton, 2012
        [2] "Generating Sequences With Recurrent Neural Networks", Graves, 2014
    """

    def __init__(self, learning_rate=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 max_grad_norm=None):  # 0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(params, lr=self.learning_rate, alpha=self.alpha, eps=self.eps,
                                           weight_decay=self.weight_decay, momentum=self.momentum,
                                           centered=self.centered)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()


class SGD(object):
    r"""Stochastic Gradient Descent

    References:
        [1] "A Stochastic Approximation Method", Robbins and Monro, 1951
        [2] "On the importance of initialization and momentum in deep learning", Sutskever et al., 2013
    """

    def __init__(self, learning_rate=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 max_grad_norm=None): #0.5
        self.optimizer = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.max_grad_norm = max_grad_norm

    def optimize(self, params, loss):
        # create optimizer if necessary
        if self.optimizer is None:
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, dampening=self.dampening,
                                       weight_decay=self.weight_decay, nesterov=self.nesterov)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()
