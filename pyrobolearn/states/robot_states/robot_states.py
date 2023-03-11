#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the basic robot states

Check also the joint, link, and sensor states.

Dependencies:
- `pyrobolearn.states`
- `pyrobolearn.robots`
"""

# import copy
from abc import ABCMeta
import numpy as np

from pyrobolearn.states import State
from pyrobolearn.robots import Robot

from pyrobolearn.utils.transformation import get_matrix_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotState(State):
    r"""Robot state (abstract)

    This class is inherited by all the states that described a robot state.
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the robot state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot")
        self._robot = robot
        super(RobotState, self).__init__(window_size=window_size, axis=axis, ticks=ticks)

    @property
    def robot(self):
        """Return the robot instance"""
        return self._robot

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        state = self.__class__(robot=robot, window_size=self.window_size, axis=self.axis, ticks=self.ticks)

        memo[self] = state
        return state


class BasePositionState(RobotState):
    r"""Base position state

    This is the state that returns the base position with respect to the world frame.
    """

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the base position state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BasePositionState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the base position state data."""
        self.data = self.robot.get_base_position()


class BaseHeightState(RobotState):
    r"""Base height state

    This is the state that returns the base height with respect to the world frame.
    """

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the base height state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BaseHeightState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the height state data."""
        self.data = np.array([self.robot.get_base_position()[-1]])


class BaseOrientationState(RobotState):
    r"""Base orientation state

    This is the state that returns the base orientation with respect to the world frame.
    """

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the base orientation state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BaseOrientationState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the orientation state data."""
        self.data = self.robot.get_base_orientation()


class BaseAxisState(RobotState):
    r"""Base axis state

    This is the state that returns the specified axis computed from the base orientation with respect to the world
    frame. From the orientation expressed as a quaternion, this is first transformed as a rotation matrix from which a
    column is then extracted (that column represents one of its axis).
    """

    def __init__(self, robot, base_axis=2, window_size=1, axis=None, ticks=1):
        """
        Initialize the base orientation state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state.
            base_axis (int): base axis index which is x=0, y=1, or z=2.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        # check basis axis
        if base_axis < 0:
            base_axis = 0
        elif base_axis > 2:
            base_axis = 2
        self._base_axis = int(base_axis)

        # call parent class
        super(BaseAxisState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the orientation state data."""
        self.data = get_matrix_from_quaternion(self.robot.get_base_orientation())[:, self._base_axis]


class BasePoseState(RobotState):
    r"""Base pose state

    This is the state that returns the base position [x,y,z] and orientation (expressed as a quaternion [x,y,z,w])
    with respect to the world frame.
    """

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the base pose state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BasePoseState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the base position state data."""
        self.data = self.robot.get_base_pose(concatenate=True)


class BaseLinearVelocityState(RobotState):
    r"""Base linear velocity state

    This is the state that returns the base linear velocity with respect to the world frame.
    """

    def __init__(self, robot, link_frame=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the base linear velocity state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state.
            link_frame (None, int): the frame in which the base linear velocity is represented. If None, it will be
                the world frame, if -1, it will be the base, and if another int, it will be the corresponding link
                frame.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BaseLinearVelocityState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the base linear velocity state data."""
        self.data = self.robot.get_base_linear_velocity()


class BaseAngularVelocityState(RobotState):
    r"""Base angular velocity state

    This is the state that returns the base angular velocity with respect to the world frame.
    """

    def __init__(self, robot, link_frame=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the base position state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state.
            link_frame (None, int): the frame in which the base angular velocity is represented. If None, it will be
                the world frame, if -1, it will be the base, and if another int, it will be the corresponding link
                frame.
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BaseAngularVelocityState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the base angular velocity state data."""
        self.data = self.robot.get_base_angular_velocity()


class BaseVelocityState(RobotState):
    r"""Base velocity state

    This is the state that returns the base linear [vx, vy, vz] and angular velocity [wx, wy, wz] with respect to the
    world frame.
    """

    def __init__(self, robot, window_size=1, axis=None, ticks=1):
        """
        Initialize the base velocity state.

        Args:
            robot (Robot): instance of Robot which allows to access to the robot state
            window_size (int): window size of the state. This is the total number of states we should remember. That
                is, if the user wants to remember the current state :math:`s_t` and the previous state :math:`s_{t-1}`,
                the window size is 2. By default, the :attr:`window_size` is one which means we only remember the
                current state. The window size has to be bigger than 1. If it is below, it will be set automatically
                to 1. The :attr:`window_size` attribute is only valid when the state is not a combination of states,
                but is given some :attr:`data`.
            axis (int, None): axis to concatenate or stack the states in the current window. If you have a state with
                shape (n,), then if the axis is None (by default), it will just concatenate it such that resulting
                state has a shape (n*w,) where w is the window size. If the axis is an integer, then it will just stack
                the states in the specified axis. With the example, for axis=0, the resulting state has a shape of
                (w,n), and for axis=-1 or 1, it will have a shape of (n,w). The :attr:`axis` attribute is only when the
                state is not a combination of states, but is given some :attr:`data`.
            ticks (int): number of ticks to sleep before getting the next state data.
        """
        super(BaseVelocityState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the base linear velocity state data."""
        self.data = self.robot.get_base_velocity(concatenate=True)
