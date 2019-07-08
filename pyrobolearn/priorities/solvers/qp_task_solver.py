#!/usr/bin/env python
r"""Provide the task solver that uses quadratic programming.

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

    ###########
    # Methods #
    ###########

    def update(self):
        """Update the priority task; compute the matrices and vectors to be used later in the `solve` method."""
        self.task.update()

    def solve(self):
        """Solve the priority task."""
        if self.task.tasks:
            for soft_task in self.task.tasks:
                As = np.vstack([np.dot(np.sqrt(task.weight), task.A) for task in soft_task])
                bs = np.vstack([np.dot(np.sqrt(task.weight), task.b) for task in soft_task])
                # x = self.solver.optimize(P=As.T.dot(As), q=-bs.T.dot(), G=, h=, A=, b=)
        else:
            pass
