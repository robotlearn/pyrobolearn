#!/usr/bin/env python
"""Define some common terminal conditions for the environment.
"""

import numpy as np

from pyrobolearn.robots import Robot
from pyrobolearn.states import LinkState

from pyrobolearn.utils.orientation import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class TerminalCondition(object):
    r"""Terminal Condition

    This class provides the basic layout for the `TerminalCondition` class and its child classes.
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


class FailedCondition(TerminalCondition):
    r"""Failed Terminal Condition

    This determines when a policy or multiple ones have failed to perform a certain task.
    """
    pass


class SucceededCondition(TerminalCondition):
    r"""Succeeded Terminal Condition

    This determines when a policy or multiple ones have succeeded to perform a certain task.
    """
    pass


class GymTerminalCondition(TerminalCondition):
    r"""OpenAI Gym Terminal Condition

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

    def __init__(self, robot, height_threshold=None, angle_threshold=np.pi/6):
        """
        Check if the robot has fallen.

        Args:
            robot (Robot): robot instance.
            height_threshold (float, None): height threshold. If the base height of the robot is below this threshold,
                it will be considered that the robot has fallen. If None, it will use the original robot base height
                and divided by 3 as the threshold.
            angle_threshold (float): angle threshold in radians. If the angle between the initial robot base up
                vector, and the current base up vector is bigger than the threshold, it will be considered that the
                robot has fallen. Normally, the initial robot base up vector points upward. By default, it is 30
                degrees (=pi/6 rad).
        """
        self.robot = robot
        self.height_threshold = height_threshold if height_threshold is not None else robot.base_height/3.
        self.angle_threshold = angle_threshold

    def _compute_angle(self):
        """Compute angle between the initial base up vector and current base up vector."""
        up_vector = get_matrix_from_quaternion(self.robot.get_base_orientation())[:, 2]
        angle = np.arccos(np.dot(self.robot.base_up_vector, up_vector))
        return angle

    def _compute_height(self):
        """Compute the current height."""
        return self.robot.get_base_position()[2]

    def check(self):
        height = self._compute_height()
        height_condition = height < self.height_threshold
        angle = self._compute_angle()
        angle_condition = angle > self.angle_threshold
        return height_condition or angle_condition

    def __repr__(self):
        description = '{} (\n\tbase_height={} ?<? height_threshold={}, \n\tangle_up_vector={} ?>? angle_threshold={}' \
                      '\n)'.format(self.__class__.__name__, self._compute_height(), self.height_threshold,
                                   self._compute_angle(), self.angle_threshold)
        return description


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