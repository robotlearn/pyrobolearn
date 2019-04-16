#!/usr/bin/env python
r"""Provide the various task solvers which uses QP.


References:
    [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.tasks.task import Task
from pyrobolearn.optimizers.qpsolvers_optimizer import QP


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TaskSolver(object):
    r"""Task solver.
    """

    def __init__(self, task):
        """
        Initialize the task solver.

        Args:
            task (Task): Priority tasks.
        """
        self.task = task
        self.solver = QP(method='qpoases')

    ##############
    # Properties #
    ##############

    @property
    def task(self):
        """Return the priority task."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the priority task."""
        if not isinstance(task, Task):
            raise TypeError("Expecting the given 'task' to be an instance of `Task`, instead got: "
                            "{}".format(type(task)))
        self._task = task

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

    #############
    # Operators #
    #############

    def __call__(self):
        return self.solve()
