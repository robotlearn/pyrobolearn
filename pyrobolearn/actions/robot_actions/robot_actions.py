#!/usr/bin/env python
"""Define the basic robot actions

Check also the joint, link, and end-effector actions.

Dependencies:
- `pyrobolearn.actions`
- `pyrobolearn.robots`
"""

from abc import ABCMeta
from pyrobolearn.actions import Action
from pyrobolearn.robots import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
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
        super(RobotAction, self).__init__()
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot")
        self._robot = robot

    @property
    def robot(self):
        return self._robot

    def isDiscrete(self):
        return False

    def isContinuous(self):
        return True
