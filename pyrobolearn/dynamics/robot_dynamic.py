#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides robot dynamic transition functions
"""

from pyrobolearn.robots.robot import Robot
from pyrobolearn.dynamics.dynamic import DynamicModel


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class PhysicalDynamicModel(DynamicModel):
    r"""Physical Dynamic Model

    Dynamic model described by mathematical/physical equations.
    """

    def __init__(self, state, action):
        super(PhysicalDynamicModel, self).__init__(state, action)


class RobotDynamicModel(PhysicalDynamicModel):
    r"""Robot Dynamical Model

    This is the mathematical model of the robots.

    Limitations:
    * mathematical assumptions such as rigid bodies
    * approximation of the dynamics
    * the states/actions have to be robot states/actions
    """

    def __init__(self, state, action, robot):
        super(RobotDynamicModel, self).__init__(state, action)
        self.robot = robot

    ##############
    # Properties #
    ##############

    @property
    def robot(self):
        """Return the robot instance."""
        return self._robot

    @robot.setter
    def robot(self, robot):
        """Set the robot instance."""
        if not isinstance(robot, Robot):
            raise TypeError("Expecting the given robot to be an instance of `Robot`, instead got: "
                            "{}".format(type(robot)))
        self._robot = robot
