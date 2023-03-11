#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the basic robot actions

Check also the joint, link, and end-effector actions.

Dependencies:
- `pyrobolearn.actions`
- `pyrobolearn.robots`
"""

import copy
from abc import ABCMeta

from pyrobolearn.actions import Action
from pyrobolearn.robots import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotAction(Action):
    """Robot Action class.

    This class defines how to map an action produced by the policy to the robot.
    Each action of this type is associated to a particular robot.
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot):
        """Initialize the abstract robot action.

        Args:
            robot (Robot): a robot instance.
        """
        super(RobotAction, self).__init__()

        # check robot instance
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot, but instead got: "
                            "{}".format(type(robot)))
        self._robot = robot

    @property
    def robot(self):
        """Return the robot instance."""
        return self._robot

    # def is_discrete(self):
    #     """By default, robot actions are continuous."""
    #     return False
    #
    # def is_continuous(self):
    #     """By default, robot actions are continuous."""
    #     return True

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        action = self.__class__(robot=robot)
        memo[self] = action
        return action
