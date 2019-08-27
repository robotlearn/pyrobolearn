#!/usr/bin/env python
r"""Provide the task solver that uses quadratic programming.

A quadratic program (QP) is written in standard form [1]_ as:

.. math::

    x^* =& \arg \min_x \; \frac{1}{2} x^T Q x + p^T x \\
    & \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = k \end{array}


where :math:`x` is the vector being optimized (in robotics, it can be joint positions, velocities, torques, ...),
"the matrix :math:`Q` and vector :math:`p` are used to define any quadratic objective function of these variables,
while the matrix-vector couples :math:`(G,h)` and :math:`(F,k)` respectively define inequality and equality
constraints" [1]_. Inequality constraints can include the lower bounds and upper bounds of :math:`x` by setting
:math:`G` to be the identity matrix or minus this one, and :math:`h` to be the upper or minus the lower bounds.

For instance, the quadratic objective function :math:`||Ax - b||_{W}^2` (where :math:`W` is a symmetric weight matrix)
is given in the standard form as:

.. math:: ||Ax - b||_{W}^2 = (Ax - b)^\top W (Ax - b) = x^\top A^\top W A x - 2 b^\top W A x + b^\top W b

where the last term :math:`b^\top W b` can be removed as it does not depend on the variables we are optimizing (i.e.
:math:`x`). We thus have :math:`Q = A^\top W A` a symmetric matrix and :math:`p = -2 A^\top W b`.

Note that if we had instead :math:`||Ax - b||_{W}^2 + c^\top x`, this could be rewritten as:

.. math:: ||Ax - b||_{W}^2 + c^\top x = x^\top A^\top W A x - (2 b^\top W A - c^\top) x + b^\top W b,

giving :math:`Q = A^\top W A` and :math:`p = (c - 2 A^\top W b)`.

Many control problems in robotics can be formulated as a quadratic programming problem. For instance, let's assume
that we want to optimize the joint velocities :math:`\dot{q}` given the end-effector's desired position and velocity
in task space. We can define the quadratic problem as:

.. math:: || J(q) \dot{q} - v_c ||^2

where :math:`v_c = K_p (x_d - x) + K_d (v_d - \dot{x})` (using PD control), with :math:`x_d` and :math:`x` the desired
and current end-effector's position respectively, and :math:`v_d` is the desired velocity. The solution to this
task (i.e. optimization problem) is the same solution given by `inverse kinematics`. Now, you can even obtain the
damped least squares inverse kinematics by adding a soft task such that
:math:`||J(q)\dot{q} - v_c||^2 + ||q||^2` is optimized (note that :math:`||q||^2 = ||A q - b||^2`, where :math:`A=I` is
the identity matrix and :math:`b=0` is the zero/null vector).


- **Soft** priority tasks: with soft-priority tasks, the quadratic programming problem being minimized for :math:`n`
  such tasks is given by:

  .. math::

      \begin{array}{c}
      x^* = \arg \min_x ||A_1 x - b_1||_{W_1}^2 + ||A_2 x - b_2 ||_{W_2}^2 + ... + ||A_n x - b_n ||_{W_n}^2 \\
      \text{subj. to } \; \begin{array}{c} Gx \leq h \\ Fx = k \end{array}
      \end{array}

  Often, the weight PSD matrices :math:`W_i` are just positive scalars :math:`w_i`. This problem can notably be solved
  by stacking the :math:`A_i` one of top of another, and stacking the :math:`b_i` and :math:`W_i` in the same manner,
  and solving :math:`||A x - b||_{W}^2`. This is known as the augmented task. When the matrices :math:`A_i` are
  Jacobians this is known as the augmented Jacobian (which can sometimes be ill-conditioned).

- **Hard** priority tasks: with hard-priority tasks, the quadratic programming problem for :math:`n` tasks is defined
  in a sequential manner, where the first most important task will be first optimized, and then the subsequent tasks
  will be optimized one after the other. Thus, the first task to be optimized is given by:

  .. math::

       x_1^* =& \arg \min_x \; ||A_1 x - b_1||^2 \\
       & \text{subj. to } \; \begin{array}{c} G_1 x \leq h_1 \\ F_1 x = k_1 \end{array}

  while the second next most important task that would be solved is given by:

  .. math::

      x_2^* =& \arg \min_x \; ||A_2 x - b_2||^2 \\
      &  \begin{array}{cc} \text{subj. to }
                & G_2 x \leq h_2 \\
                & F_2 x = k_2 \\
                & A_1 x = A_1 x_1^* \\
                & G_1 x \leq h_1 \\
                & F_1 x = k_1,
        \end{array}

  until the :math:`n` most important task, given by:

  .. math::

      x_n^* =& \arg \min_x  \; ||A_n x - b_n||^2 \\
      &  \begin{array}{cc} \text{subj. to } & A_1 x = A_1 x_1^* \\
                & ... \\
                & A_{n-1} x = A_{n-1} x_{n-1}^* \\
                & G_1 x \leq h_1 \\
                & ... \\
                & G_n x \leq h_n \\
                & F_1 x = k_1 \\
                & ... \\
                & F_n x = k_n. \end{array}

  By setting the previous :math:`A_{i-1} x = A_{i-1} x_{i-1}^*` as equality constraints, the current solution
  :math:`x_i^*` won't change the optimality of all higher priority tasks.

References:
    - [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.solvers.task_solver import TaskSolver
from pyrobolearn.optimizers.qpsolvers_optimizer import QP


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi, C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class QPTaskSolver(TaskSolver):
    r"""QP Task Solver.

    The QP task solver uses QP to solve a task or stack of tasks.
    """

    def __init__(self, task, method='quadprog'):
        """
        Initialize the task solver.

        Args:
            task (Task): Priority tasks.
            method (str): QP method/library to use. Select between ['cvxopt', 'cvxpy', 'ecos', 'gurobi', 'mosek',
              'osqp', 'qpoases', 'quadprog']
        """
        solver = QP(method=method)
        super(QPTaskSolver, self).__init__(task, solver)

    ##############
    # Properties #
    ##############

    @property
    def solver(self):
        """Return the optimizer/solver instance."""
        return self._solver

    @solver.setter
    def solver(self, solver):
        """Set the optimizer/solver instance."""
        if solver is not None and not isinstance(solver, QP):
            raise TypeError("Expecting the given 'solver' to be an instance of `QP`, instead got: "
                            "{}".format(type(solver)))
        self._solver = solver

    ###########
    # Methods #
    ###########

    def update(self):
        """Update the priority task; compute the matrices and vectors to be used later in the `solve` method."""
        self.task.update()

    def solve(self, x0=None):
        """Solve the priority task.

        Args:
            x0 (np.array[float[N]], None): initial guess for the optimized variables.

        Returns:
            np.array[float[N]]: the optimized variables.
        """
        # get task objectives and constraints
        # objectives
        As = self.task.A
        bs = self.task.b
        cs = self.task.c
        Ws = self.task.W
        # constraints
        Gs = self.task.G
        hs = self.task.h
        Fs = self.task.F
        ks = self.task.k

        # if not a stack of tasks, just transform the task objectives/constraints into lists
        if not self.task.is_stack_of_tasks():
            As, bs, cs, Ws = [As], [bs], [cs], [Ws]
            Gs, hs, Fs, ks = [Gs], [hs], [Fs], [ks]

        # solve
        x_opt, x_opts, x_projs = x0, [], []
        for i in range(len(As)):
            var = As[i].T.dot(Ws[i])
            Q = var.dot(As[i])
            p = cs[i] - var.dot(bs[i])
            G = np.concatenate(Gs[:i+1])
            h = np.concatenate(hs[:i+1])
            F = np.concatenate(Fs[:i+1] + As[:i])
            k = np.concatenate(ks[:i+1] + x_projs[:i])

            # solve (by starting from previous optimized solution)
            x_opt = self.solver.optimize(Q=Q, p=p, x0=x_opt, G=G, h=h, A=F, b=k)
            x_opts.append(x_opt)

            # project best solution (will be used later in the stack for constraints)
            if i < len(As) - 1:
                x_proj = As[i].dot(x_opt)
                x_projs.append(x_proj)

        # return optimized variables
        return x_opt
