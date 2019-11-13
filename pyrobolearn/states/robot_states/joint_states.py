#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

import copy
from abc import ABCMeta
import numpy as np
from gym import spaces

from pyrobolearn.states.robot_states.robot_states import RobotState, Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointState(RobotState):
    r"""Joint State of a robot (abstract class).
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the joint state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        # check if robot instance
        if not isinstance(robot, Robot):
            raise TypeError("The 'robot' parameter has to be an instance of Robot")

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.get_joint_ids()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

        super(JointState, self).__init__(robot, window_size=window_size, axis=axis, ticks=ticks)

    def __copy__(self):
        """Return a shallow copy of the state. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints, window_size=self.window_size, axis=self.axis,
                              ticks=self.ticks)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the state. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joint_ids = copy.deepcopy(self.joints)
        state = self.__class__(robot=robot, joint_ids=joint_ids, window_size=self.window_size, axis=self.axis,
                               ticks=self.ticks)

        memo[self] = state
        return state


class JointPositionState(JointState):
    r"""Joint Position State

    Return the joint positions as the state.
    """

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the joint position state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        super(JointPositionState, self).__init__(robot, joint_ids, window_size=window_size, axis=axis, ticks=ticks)

        # define the space based on the joint type  # TODO

    def _read(self):
        """Read the next joint position state."""
        self.data = self.robot.get_joint_positions(self.joints)

    def _reset(self):
        """Reset the state."""
        # reset counter
        self.cnt = 0.

        # reset the robot joint position based on the data
        if len(self.data) > 0:
            self.robot.reset_joint_states(q=self.data[0], joint_ids=self.joints)

        # read the next data
        self._read()


class JointTrigonometricPositionState(JointState):
    r"""Joint Trigonometric Position State

    Return the trigonometric joint positions as the state. That is, it returns
    :math:`[\cos(q_1), \sin(q_1), ..., \cos(q_n), \sin(q_n)]`. All the values are between -1 and 1.
    """

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the trigonometric joint position state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        super(JointTrigonometricPositionState, self).__init__(robot, joint_ids, window_size=window_size, axis=axis,
                                                              ticks=ticks)

        high = np.ones(2 * len(self.joints))
        self._space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _read(self):
        """Read the next joint position state."""
        q = self.robot.get_joint_positions(self.joints)
        self.data = np.vstack((np.cos(q), np.sin(q))).T.reshape(-1)


class JointVelocityState(JointState):
    r"""Joint Velocity State

    Return the joint velocities as the state.
    """

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the joint velocity state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        super(JointVelocityState, self).__init__(robot, joint_ids, window_size=window_size, axis=axis, ticks=ticks)

        # define space
        max_vel = self.robot.get_joint_max_velocities(joint_ids=self.joints)
        if np.allclose(max_vel, 0):
            print("WARNING: joint max velocities are 0, setting low=-10 and high=10.")
            max_vel = 10 * np.ones(len(self.joints))  # TODO: np.infty instead of 10?
        self._space = spaces.Box(low=-max_vel, high=max_vel, dtype=np.float32)

    def _read(self):
        """Read the next joint velocity state."""
        self.data = self.robot.get_joint_velocities(self.joints)

    def _reset(self):
        """Reset the state."""
        # reset counter
        self.cnt = 0.

        # reset the robot joint position based on the data
        if len(self.data) > 0:
            print("reset data: ", self.data[0])
            self.robot.reset_joint_states(dq=self.data[0], joint_ids=self.joints)

        # read the next data
        self._read()


class JointForceTorqueState(JointState):
    r"""Joint Force Torque State

    Return the joint force and torques as the state.
    """

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the joint force torque state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        super(JointForceTorqueState, self).__init__(robot, joint_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next joint force torque state."""
        self.data = self.robot.get_joint_torques(self.joints)


class JointAccelerationState(JointState):
    r"""Joint Acceleration State.

    Return the joint accelerations as the state. In order to produce the joint accelerations, we first read the
    joint torques and then applied forward dynamics to get the corresponding joint accelerations.
    """

    def __init__(self, robot, joint_ids=None, window_size=1, axis=None, ticks=1):
        """
        Initialize the joint acceleration state.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, int[N]): joint id or list of joint ids.
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
        super(JointAccelerationState, self).__init__(robot, joint_ids, window_size=window_size, axis=axis, ticks=ticks)

    def _read(self):
        """Read the next joint acceleration state."""
        self.data = self.robot.get_joint_accelerations(self.joints)
