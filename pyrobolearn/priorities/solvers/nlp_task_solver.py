#!/usr/bin/env python
r"""Provide the non-linear task solver.

Warnings: This optimization process might take time to solve the task.

References:
    - [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.solvers.task_solver import TaskSolver
from pyrobolearn.optimizers.nlopt_optimizer import NLopt


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class NLPTaskSolver(TaskSolver):
    r"""Nonlinear Programming Task Solver.

    The NLP task solver uses Non-Linear Programming to solve a task or stack of tasks.

    Warnings: This optimization process might take time to solve the task.
    """

    def __init__(self, task, method, submethod=None, seed=None):
        """
        Initialize the task solver.

        Args:
            task (Task): Priority tasks.
            method (str): primary optimization method to be used.
            submethod (str): sub-optimization method to be used in the primary optimization method.
            seed (None, int): random seed
        """
        solver = NLopt(method=method, submethod=submethod, seed=seed)
        super(NLPTaskSolver, self).__init__(task, solver)

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
