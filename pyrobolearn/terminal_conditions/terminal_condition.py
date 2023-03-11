#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define some common terminal conditions for the environment.
"""

import copy
import numpy as np

from pyrobolearn.robots import Robot
from pyrobolearn.states import LinkState

from pyrobolearn.utils.transformation import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
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

    def __init__(self, btype=None, name=None):
        """
        Initialize the terminal condition.

        Args:
            btype (bool, str, None): if the terminal condition represents a failure or success condition. If None, it
                represents a neutral terminal condition (which is neither a failure or success condition, but just
                means the episode is over). If string, it has to be among {"success", "failure", "neutral"}.
            name (str): name of the final condition
        """
        self.btype = btype
        self._over = False
        self._achieved = False
        self.name = name

    ##############
    # Properties #
    ##############

    @property
    def btype(self):
        """Return if the terminal condition is a neutral (None), a failure (False) or a success (True) condition."""
        return self._btype

    @btype.setter
    def btype(self, value):
        """Set the success variable."""
        if value is not None and not isinstance(value, bool) and not isinstance(value, str):
            raise TypeError("Expecting the given 'btype' variable to be a boolean, string, or None, but got instead: "
                            "{}".format(type(value)))
        if isinstance(value, str):
            if value == 'neutral':
                value = None
            elif value == 'failure':
                value = False
            elif value == 'success':
                value = True
            else:
                raise ValueError("Expecting the given string to be among ['success', 'failure', 'neutral'], but "
                                 "instead got: {}".format(value))
        self._btype = value

    ###########
    # Methods #
    ###########

    def is_over(self):
        """Return if the condition is over or not."""
        return self._over

    def succeeded(self):
        """Return if the condition succeeded or not."""
        return self._achieved

    def type(self):
        """Return if it a success, failure, or neutral terminal condition."""
        return self._btype

    def type_str(self):
        """Return the string representing the type of terminal condition."""
        if self._btype is None:
            return "neutral"
        if self._btype:
            return "success"
        return "failure"

    def reset(self):
        """
        Reset the terminal condition.
        """
        self._over = False

    def check(self):
        """
        Check if the terminating condition has been fulfilled, and return True or False accordingly
        """
        return self._over

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the terminal condition."""
        return self.__class__.__name__

    def __call__(self):
        """Check the terminal condition."""
        return self.check()

    def __bool__(self):
        """Return a bool based on the terminating condition."""
        return self.check()

    __nonzero__ = __bool__


# class FailedCondition(TerminalCondition):
#     r"""Failed Terminal Condition
#
#     This determines when a policy or multiple ones have failed to perform a certain task.
#     """
#     pass
#
#
# class NeutralCondition(TerminalCondition):
#     r"""Neutral Terminal Condition
#
#     This determines when an environment has ended.
#     """
#     pass
#
#
# class SucceededCondition(TerminalCondition):
#     r"""Succeeded Terminal Condition
#
#     This determines when a policy or multiple ones have succeeded to perform a certain task.
#     """
#     pass


class HasFallenCondition(TerminalCondition):
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
        super(HasFallenCondition, self).__init__(btype='failure')
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

    def __str__(self):
        """Return a string describing the terminal condition."""
        description = '{} (\n\tbase_height={} ?<? height_threshold={}, \n\tangle_up_vector={} ?>? angle_threshold={}' \
                      '\n)'.format(self.__class__.__name__, self._compute_height(), self.height_threshold,
                                   self._compute_angle(), self.angle_threshold)
        return description

    def __copy__(self):
        """Return a shallow copy of the terminal condition. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, height_threshold=self.height_threshold,
                              angle_threshold=self.angle_threshold)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the terminal condition. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)
        terminal = self.__class__(robot=robot, height_threshold=self.height_threshold,
                                  angle_threshold=self.angle_threshold)
        memo[self] = terminal
        return terminal


class HasReachedCondition(TerminalCondition):
    r"""Has Reached Condition

    Check if the robot or a part of it has reached a certain position, configuration, or state for a certain amount
    of time/steps.
    """

    def __init__(self):
        super(HasReachedCondition, self).__init__(btype='success')


class LinkInSpecifiedDirection(HasReachedCondition):
    r"""Check if the specified link is in certain direction for a certain amount of time steps.

    Specifically, it checks if the position of the link with respect to the world or another link is in a certain
    direction for a certain amount of time steps. The position vector is considered to be in the good direction if
    it belongs to the specified cone domain.
    """

    def __init__(self, state, direction, domain=(0.95, 1.), total_steps=0):
        super(LinkInSpecifiedDirection, self).__init__()
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
        pos = self.state.data[0]
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

    def __str__(self):
        """Return a string describing the terminal condition."""
        return self.__class__.__name__ + '(direction=' + str(self.direction) + ')'

    def __copy__(self):
        """Return a shallow copy of the terminal condition. This can be overridden in the child class."""
        return self.__class__(state=self.state, direction=self.direction, domain=self.domain,
                              total_steps=self.total_steps)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the terminal condition. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        state = copy.deepcopy(self.state, memo)
        direction = copy.deepcopy(self.direction)
        domain = copy.deepcopy(self.domain)
        terminal = self.__class__(state=state, direction=direction, domain=domain, total_steps=self.total_steps)
        memo[self] = terminal
        return terminal
