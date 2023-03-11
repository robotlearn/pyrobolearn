#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Provide the abstract task solver class from which all the other task solvers inherit from.

A task solver accepts as inputs a task (or a stack of tasks) and an optimizer to use to solve the task.

References:
    - [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    - [2] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np

from pyrobolearn.priorities.tasks.task import Task
from pyrobolearn.optimizers import Optimizer


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi, C++)", "Brian Delhaisse (Python + doc)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TaskSolver(object):
    r"""Task solver.

    The task solver accepts a task, or stack of tasks, and an optimization solver.
    """

    def __init__(self, task, solver=None):
        """
        Initialize the task solver.

        Args:
            task (Task): Priority tasks.
            solver (Optimizer, None): inner solver/optimizer that is used to solve the tasks.
        """
        self.task = task
        self.solver = solver

    ##############
    # Properties #
    ##############

    @property
    def task(self):
        """Return the priority task / stack of tasks."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the priority task / stack of tasks."""
        if not isinstance(task, Task):
            raise TypeError("Expecting the given 'task' to be an instance of `Task`, instead got: "
                            "{}".format(type(task)))
        self._task = task

    @property
    def solver(self):
        """Return the optimizer/solver instance."""
        return self._solver

    @solver.setter
    def solver(self, solver):
        """Set the optimizer/solver instance."""
        if solver is not None and not isinstance(solver, Optimizer):
            raise TypeError("Expecting the given 'solver' to be an instance of `Optimizer`, instead got: "
                            "{}".format(type(solver)))
        self._solver = solver

    ###########
    # Methods #
    ###########

    def update(self):
        """Update the priority task; compute the matrices and vectors to be used later in the `solve` method."""
        self.task.update()

    def solve(self):
        """Solve the priority task.

        Returns:
            np.array[float[N]]: the optimized variables.
        """
        pass

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the task solver."""
        return self.__class__.__name__

    def __call__(self):
        """Solve the task using the optimizer, and returned the optimized variables."""
        return self.solve()
