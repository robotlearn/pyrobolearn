#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Transform a given equality constraint into a task.

An equality constraint specified by :math:`Fx = k` is transformed to a soft task :math:`||Ax - b||^2`, where
:math:`A = F` and :math:`b = k`. This allows for the equality constraint to be lightly violated; by specifying the
weight :math:`W` we can specify how much the constraint should be satisfied.
"""

import pyrobolearn as prl
from pyrobolearn.priorities.constraints import EqualityConstraint


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ConstraintFromTask(EqualityConstraint):
    r"""Equality Constraint from Task

    A soft task :math:`||Ax - b||^2` is transformed to an equality constraint specified by :math:`Fx = k`, where
    :math:`F = A` and :math:`k = b`. This allows a priori for the task to not be violated, however this might results
    in an impossible task to solve.
    """

    def __init__(self, task):
        """
        Initialize the task.

        Args:
            task (Task): task.
        """
        # set task
        self.task = task

        # call superclass
        super(ConstraintFromTask, self).__init__(model=task.model)

        # first update
        self.update()

    ##############
    # Properties #
    ##############

    @property
    def task(self):
        """Get the task."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the task."""
        if not isinstance(task, prl.priorities.tasks.Task):
            raise TypeError("Expecting the given 'task' to be an instance of `Task`, but instead got: {}".format(task))
        self._task = task

    ###########
    # Methods #
    ###########

    def _update(self):
        """
        Update the task by computing the A matrix and b vector that will be used by the task solver.
        """
        # update task
        self.task.update()

        # update A_eq and b_eq
        self._A_eq = self.task.A
        self._b_eq = self.task.b
