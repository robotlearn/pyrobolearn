#!/usr/bin/env python
"""Provide the task controller.
"""

from pyrobolearn.controllers.controller import Controller
from pyrobolearn.priorities import TaskSolver

__author__ = ["Brian Delhaisse"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TaskController(Controller):
    r"""Task Controller.

    The task controller accepts a priority task, and execute it.
    """

    def __init__(self, solver, rate=1):
        """
        Initialize the task controller.

        Args:
            solver (TaskSolver): the task solver.
            rate (int, float): rate (float) at which the controller operates if we are operating in real-time. If we
                are stepping deterministically in the simulator, it represents the number of ticks (int) to sleep
                before executing the model.
        """
        super(TaskController, self).__init__()
        self.solver = solver
        self.cnt = 0
        self.rate = rate
        self.x = None

    ##############
    # Properties #
    ##############

    @property
    def solver(self):
        """Return the task solver."""
        return self._solver

    @solver.setter
    def solver(self, solver):
        """Set the task solver."""
        if not isinstance(solver, TaskSolver):
            raise TypeError("Expecting the given 'solver' to be an instance of `TaskSolver`, instead got: "
                            "{}".format(type(solver)))
        self._solver = solver

    @property
    def task(self):
        """Return the priority task."""
        return self.solver.task

    @property
    def robot(self):
        """Return the robot."""
        return self.solver.task.model

    ###########
    # Methods #
    ###########

    def compute(self, *args, **kwargs):
        # update the task
        self.solver.update()
        # solve the task
        x = self.solver.solve()
        # return the optimal vector
        return x

    def act(self, *args, **kwargs):
        if (self.cnt % self.rate) == 0:
            # compute optimal variables
            self.x = self.compute(*args, **kwargs)

        # set variables using the model
        # TODO

        # update counter
        self.cnt += 1

        return self.x
