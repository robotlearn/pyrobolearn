#!/usr/bin/env python
"""Define some common terminating condition for the environment.
"""

import numpy as np

from pyrobolearn.robots import Robot
from pyrobolearn.states import LinkState

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TerminatingCondition(object):
    r"""Terminating Condition

    This class provides the basic layout for the `TerminatingCondition` class and its child classes.
    This one can be used to check when an environment has fulfilled certain conditions and can be terminated.

    This class can be further subdivided into two categories: failed and succeeded conditions.
    * Failed conditions determine when a policy has failed to perform a certain task
    * Succeeded conditions determine when a policy has succeeded to perform a certain task

    In order to compute this condition, they can access to more information than the state provided to the agent(s).
    """
    # TODO: we should be able to combine different conditions using 'OR' and 'AND'

    def check(self):
        """
        Check if the terminating condition has been fulfilled, and return True or False accordingly
        """
        return False

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.check()

    def __bool__(self):
        return self.check()

    __nonzero__ = __bool__


class FailedCondition(TerminatingCondition):
    r"""Failed Terminating Condition

    This determines when a policy or multiple ones have failed to perform a certain task.
    """
    pass


class SucceededCondition(TerminatingCondition):
    r"""Succeeded Terminating Condition

    This determines when a policy or multiple ones have succeeded to perform a certain task.
    """
    pass


class GymTerminatingCondition(TerminatingCondition):
    r"""OpenAI Gym Terminating Condition

    Returns if the OpenAI Gym environment has terminated. This does not provide any information if the environment
    terminated because the policy succeeded or failed to perform the task.
    """

    def __init__(self, done=False):
        self.done = done

    def check(self):
        return self.done


class HasFallen(FailedCondition):
    r"""Has Fallen Condition

    Check if the given robot has fallen, by checking if its base is below a certain threshold.
    """

    def __init__(self, robot, threshold=None):
        self.robot = robot
        self.threshold = threshold if threshold is not None else robot.height/4.

    def check(self):
        return self.robot.getBasePosition()[2] < self.threshold

    def __repr__(self):
        return self.__class__.__name__ + '(threshold=' + str(self.threshold) + ')'


class HasReached(SucceededCondition):
    r"""Has Reached Condition

    Check if the robot or a part of it has reached a certain position, configuration, or state for a certain amount
    of time/steps.
    """
    pass


class LinkInSpecifiedDirection(HasReached):
    r"""Check if the specified link is in certain direction for a certain amount of time steps.

    Specifically, it checks if the position of the link with respect to the world or another link is in a certain
    direction for a certain amount of time steps. The position vector is considered to be in the good direction if
    it belongs to the specified cone domain.
    """

    def __init__(self, state, direction, domain=(0.95, 1.), total_steps=0):
        if not isinstance(state, LinkState):
            raise TypeError("Expecting the state to be an instance of LinkState, instead got: {}".format(type(state)))
        self.state = state
        self.direction = self.normalize(np.array(direction))
        self.total_steps = total_steps
        if len(domain) != 2:
            raise ValueError("Expecting the domain to be a tuple or list of 2 values.")
        self.domain = np.array(domain)
        self.cnt = 0

    @staticmethod
    def normalize(x):
        """
        Normalize the given vector.
        """
        if np.allclose(x, 0):
            return x
        return x / np.linalg.norm(x)

    def check(self):
        """
        Check if the position of the link is in certain direction between the specified interval for a certain
        amount of time steps.

        Returns:
            bool: True if the condition is satisfied.
        """
        pos = self.state._data
        pos = self.normalize(pos)
        value = np.dot(pos, self.direction)
        # check if the direction belongs to the cone domain
        if self.domain[0] <= value <= self.domain[1]:
            self.cnt += 1
            # if we are in the cone domain for a certain amount of steps
            if self.cnt > self.total_steps:
                return True
        else:
            self.cnt = 0
        return False

    def __repr__(self):
        return self.__class__.__name__ + '(direction=' + str(self.direction) + ')'