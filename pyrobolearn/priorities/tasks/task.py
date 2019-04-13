#!/usr/bin/env python
"""Provide the various tasks (i.e. objective functions) used in QP.
"""

import numpy as np

from pyrobolearn.priorities.constraints.constraint import Constraint

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: take into account constraints
# TODO: take into account hard priority tasks
class Task(object):
    r"""Task (abstract) class.

    Python implementation of Tasks based on the slides of the OpenSoT framework [1].

    References:
        [1] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN"
            ([code](https://opensot.wixsite.com/opensot),
            [slides](https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA),
            [tutorial video](https://www.youtube.com/watch?v=yFon-ZDdSyg),
            [old code](https://github.com/songcheng/OpenSoT)), Rocchi et al., 2015
    """

    def __init__(self, tasks=[], model=None, weight=1., constraints=[]):
        """
        Initialize the task.

        Args:
            tasks (list of list of Task): list of list of tasks, where the list is ordered by hard priorities, and the
                nested list contains tasks which
            model (Robot, str): robot model. If str, it needs to be the path to the URDF.
            constraints (list of Constraint): list of constraints.
        """
        self._tasks = tasks
        self._model = model
        self.weight = weight
        self._constraints = []

    ##############
    # Properties #
    ##############

    @property
    def tasks(self):
        """Return the tasks."""
        return self._tasks

    @property
    def level(self):
        """Return the level of the tree."""
        if self._tasks:
            return len(self._tasks)
        else:
            return 1

    @property
    def model(self):
        """Return the robot model."""
        return self._model

    @property
    def weight(self):
        """Return the relative weight (used for soft priorities)."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        if not isinstance(weight, (int, float)):
            raise TypeError("Expecting the relative weight to be an int or float, instead got: {}".format(type(weight)))
        if weight < 0:
            raise ValueError("Expecting the relative weight to be positive.")
        self._weight = weight

    @property
    def constraints(self):
        """Return the constraints."""
        return self._constraints

    ###########
    # Methods #
    ###########

    def _compute(self):  # update
        """Compute the task.

        Returns:
            np.array: A matrix used in QP.
            np.array: b vector used in QP.
        """
        pass

    def compute(self):  # update
        if self.tasks:
            for hard_task in self.tasks:
                results = [soft_task.compute() for soft_task in hard_task]
                As = np.vstack([result[0] for result in results])
                bs = np.vstack([result[1] for result in results])
            # TODO: continue for hard priority tasks
            return As, bs
        return self._compute()

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a string representing the class."""
        if self.tasks:
            tasks = []
            for i, soft_tasks in enumerate(self.tasks):
                results = []
                for task in soft_tasks:
                    if task.weight == 1:
                        results.append(str(task))
                    else:
                        results.append(str(task.weight) + ' * ' + str(task))
                soft_task = ' + '.join(results)
                tasks.append('Priority {}: '.format(i+1) + soft_task)
            return '\n'.join(tasks)
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__repr__()

    def __call__(self):
        return self.compute()

    def __add__(self, other):  # TODO: check when other has some tasks
        """Add a soft priority task."""
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        if len(self.tasks) > 0:
            tasks = list(self.tasks)
            tasks[-1].append(other)
            return Task(tasks=tasks)
        return Task(tasks=[[self, other]])

    def __div__(self, other):
        """Add a hard priority task."""
        if not isinstance(other, Task):
            raise TypeError("Expecting 'other' to be an instance of Task, instead got: {}".format(type(other)))
        tasks = list(self.tasks)
        if other.tasks:
            # append all the other tasks
            for task in other.tasks:
                tasks.append(task)
        else:
            tasks.append([other])
        return Task(tasks=tasks)

    def __lshift__(self, other):
        """Insert a constraint (in-place operation)."""
        if not isinstance(other, Constraint):
            raise TypeError("Expecting 'other' to be an instance of Constraint, instead got: {}".format(type(other)))
        self._constraints.append(other)

    def __mul__(self, other):
        """Multiply the task by a relative weight."""
        if not isinstance(other, (int, float)) or other < 0:
            raise TypeError("Expecting a positive integer or float for the weight.")
        self.weight = other

    def __rmul__(self, other):
        self.__mul__(other)


# Tests
if __name__ == '__main__':
    task1 = Task(weight=2)
    task2 = Task(weight=3)
    task = Task(tasks=[[task1, task2], [task1]])
    print(task)
