#!/usr/bin/env python
"""Define the various joint states

This includes notably the joint positions, velocities, and force/torque states.
"""

from abc import ABCMeta

from pyrobolearn.states.robot_states.robot_states import RobotState, Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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

    def _read(self):
        """Read the next joint position state."""
        self.data = self.robot.get_joint_positions(self.joints)


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

    def _read(self):
        """Read the next joint velocity state."""
        self.data = self.robot.get_joint_velocities(self.joints)


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
