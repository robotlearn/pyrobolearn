# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide a wrapper around quadratic programming solvers.

References:
    [1] https://github.com/stephane-caron/qpsolvers
"""

import numpy as np

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

from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


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

class QP(Optimizer):
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

    def __init__(self, method='quadprog', *args, **kwargs):
        """
        Initialize the QP solver.

        Args:
            method (str): QP method/library to use. Select between ['cvxopt', 'cvxpy', 'ecos', 'gurobi', 'mosek',
                'osqp', 'qpoases', 'quadprog']
        """
        super(QP, self).__init__(*args, **kwargs)

        solvers = set(qpsolvers.available_solvers)
        if len(solvers) == 0:
            raise ValueError("No QP solvers have been found on this computer. Please install one of the QP modules")
        if method not in solvers:
            method = 'quadprog'
        self.method = method

        # check methods that require a symmetric matrix for P
        methods = ['cvxopt', 'osqp', 'quadprog']
        self.sym_proj = True if self.method in set(methods) else False

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_symmetric(X, tol=1e-8):
        """Check if the given matrix is symmetric."""
        return np.allclose(X, X.T, atol=tol)

    ###########
    # Methods #
    ###########

    def optimize(self, Q, p, x0=None, G=None, h=None, A=None, b=None):
        r"""
        Optimize the given quadratic problem.

        .. math::

            \min_{x \in \mathbb{R}^N} \frac{1}{2} x^T Q x + p^T x

        subject to

        .. math::

            Gx \leq h
            Ax = b

        Args:
            Q (np.array[N,N]): matrix used in the QP objective function where `N` is the size of the vector `x` being
                optimized.
            p (np.array[N]): vector used in the QP objective function where `N` is the size of the vector `x` being
                optimized.
            G (np.array[M,N]): matrix used in the inequality constraint, where `M` is the number of inequalities, and
                `N` is the size of the vector `x` being optimized. Note that if you have lower and upper bounds for
                the vector `x`, you can set :attr:`G` to be the concatenation of :math:`[-I, I]^\top`, where :math:`I`
                is the identity matrix.
            h (np.array[M]): vector used in the inequality constraint, where `M` is the number of inequalities. Note
                that if you have lower and upper bounds for the vector `x`, you can set :attr:`h` to be the
                concatenation of :math:`[-b_l^\top, b_u^\top]`, where :math:`b_l` and :math:`b_u` are the lower and
                upper bounds respectively.
            A (np.array[K,N]): matrix used in the equality constraint.
            b (np.array[K,N]): vector used in the equality constraint.

        Returns:
            np.array: QP solution
        """
        return qpsolvers.solve_qp(Q, p, G, h, A, b, solver=self.method, initvals=x0, sym_proj=self.sym_proj)
