#!/usr/bin/env python
"""Define the basic robot states

Check also the joint, link, and sensor states.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.robots`
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from pyrobolearn.states import State
from pyrobolearn.robots import Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "(c) Brian Delhaisse"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotState(State):
    r"""Robot state (abstract)

    This class is inherited by all the states that described a robot state.
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot):
        """
        Initialize the robot state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
        """
        super(RobotState, self).__init__()
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot")
        self._robot = robot

    @property
    def robot(self):
        """Return the robot instance"""
        return self._robot

    @abstractmethod
    def _read(self):
        pass


class BasePositionState(RobotState):
    r"""Base position state

    This is the state that returns the base position with respect to the world frame.
    """

    def __init__(self, robot):
        super(BasePositionState, self).__init__(robot)
        self._read()

    def _read(self):
        self._data = self.robot.getBasePosition()


class BaseHeightState(RobotState):
    r"""Base height state

    This is the state that returns the base height with respect to the world frame.
    """

    def __init__(self, robot):
        super(BaseHeightState, self).__init__(robot)
        self._read()

    def _read(self):
        self._data = np.array([self.robot.getBasePosition()[-1]])


class BaseOrientationState(RobotState):
    r"""Base orientation state

    This is the state that returns the base orientation with respect to the world frame.
    """

    def __init__(self, robot):
        super(BaseOrientationState, self).__init__(robot)
        self._read()

    def _read(self):
        self._data = self.robot.getBaseOrientation(convert_to_numpy_quaternion=False)


class BaseLinearVelocityState(RobotState):
    r"""Base linear velocity state

    This is the state that returns the base linear velocity with respect to the world frame.
    """

    def __init__(self, robot):
        super(BaseLinearVelocityState, self).__init__(robot)
        self._read()

    def _read(self):
        self._data = self.robot.getBaseLinearVelocity()


class BaseAngularVelocityState(RobotState):
    r"""Base angular velocity state

    This is the state that returns the base angular velocity with respect to the world frame.
    """

    def __init__(self, robot):
        super(BaseAngularVelocityState, self).__init__(robot)
        self._read()

    def _read(self):
        self._data = self.robot.getBaseAngularVelocity()
