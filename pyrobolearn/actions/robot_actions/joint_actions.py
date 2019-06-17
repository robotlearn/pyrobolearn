#!/usr/bin/env python
"""Define the various joint actions

This includes notably the joint positions, velocities, and force/torque actions.
"""

import copy
import numpy as np
from abc import ABCMeta

from pyrobolearn.actions.robot_actions.robot_actions import RobotAction


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class JointAction(RobotAction):
    r"""Joint Action
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, joint_ids=None):
        """
        Initialize the joint action.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N]): joint id or list of joint ids
        """
        super(JointAction, self).__init__(robot)

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.get_joint_ids()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

    # @property
    # def size(self):
    #     return len(self.joints)

    def bounds(self):
        """Return the joint limits."""
        return self.robot.get_joint_limits(self.joints)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joints = copy.deepcopy(self.joints)
        action = self.__class__(robot=robot, joint_ids=joints)
        memo[self] = action
        return action


class JointPositionAction(JointAction):
    r"""Joint Position Action

    Set the joint positions using position control.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        self.kp, self.kd, self.max_force = kp, kd, max_force
        super(JointPositionAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_positions(self.joints)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_joint_positions(data, self.joints, kp=self.kp, kd=self.kd, forces=self.max_force)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints, kp=self.kp, kd=self.kd, max_force=self.max_force)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joints = copy.deepcopy(self.joints)
        kp = copy.deepcopy(self.kp)
        kd = copy.deepcopy(self.kd)
        max_force = copy.deepcopy(self.max_force)
        action = self.__class__(robot=robot, joint_ids=joints, kp=kp, kd=kd, max_force=max_force)
        memo[self] = action
        return action


class JointVelocityAction(JointAction):
    r"""Joint Velocity Action

    Set the joint velocities using velocity control.
    """

    def __init__(self, robot, joint_ids=None):
        super(JointVelocityAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_velocities(self.joints)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_joint_velocities(data, self.joints)


class JointPositionAndVelocityAction(JointAction):
    r"""Joint position and velocity action

    Set the joint position using position control using PD control, where the contraint error to be minimized is
    given by: :math:`error = kp * (q^* - q) - kd * (\dot{q}^* - \dot{q})`.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        super(JointPositionAndVelocityAction, self).__init__(robot, joint_ids)
        self.kp, self.kd, self.max_force = kp, kd, max_force
        pos, vel = robot.get_joint_positions(self.joints), robot.get_joint_velocities(self.joints)
        self.data = np.concatenate((pos, vel))
        self.idx = len(pos)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_joint_positions(data[:self.idx], self.joints, kp=self.kp, kd=self.kd,
                                       velocities=data[self.idx:], forces=self.max_force)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints, kp=self.kp, kd=self.kd, max_force=self.max_force)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joints = copy.deepcopy(self.joints)
        kp = copy.deepcopy(self.kp)
        kd = copy.deepcopy(self.kd)
        max_force = copy.deepcopy(self.max_force)
        action = self.__class__(robot=robot, joint_ids=joints, kp=kp, kd=kd, max_force=max_force)
        memo[self] = action
        return action


# class JointPositionVelocityAccelerationAction(JointAction):
#     r"""Set the joint positions, velocities, and accelerations.
#
#     Set the joint positions, velocities, and accelerations by computing the necessary torques / forces using inverse
#     dynamics.
#     """
#     pass


class JointForceAction(JointAction):
    r"""Joint Force Action

    Set the joint force/torque using force/torque control.
    """

    def __init__(self, robot, joint_ids=None, f_min=-np.infty, f_max=np.infty):
        super(JointForceAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_torques(self.joints)
        self.f_min = f_min
        self.f_max = f_max

    def _write(self, data):
        """apply the action data on the robot."""
        data = np.clip(data, self.f_min, self.f_max)
        self.robot.set_joint_torques(data, self.joints)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints, f_min=self.f_min, f_max=self.f_max)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joints = copy.deepcopy(self.joints)
        f_min = copy.deepcopy(self.f_min)
        f_max = copy.deepcopy(self.f_max)
        action = self.__class__(robot=robot, joint_ids=joints, f_min=f_min, f_max=f_max)
        memo[self] = action
        return action


class JointAccelerationAction(JointAction):
    r"""Joint Acceleration Action

    Set the joint accelerations using force/torque control. In order to produce the given joint accelerations,
    we use inverse dynamics which given the joint accelerations produce the corresponding joint forces/torques
    to be applied.
    """

    def __init__(self, robot, joint_ids=None, a_min=-np.infty, a_max=np.infty):
        super(JointAccelerationAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_accelerations(self.joints)
        self.a_min = a_min
        self.a_max = a_max

    def _write(self, data):
        """apply the action data on the robot."""
        data = np.clip(data, self.a_min, self.a_max)
        self.robot.set_joint_accelerations(data, self.joints)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(robot=self.robot, joint_ids=self.joints, a_min=self.a_min, a_max=self.a_max)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        joints = copy.deepcopy(self.joints)
        a_min = copy.deepcopy(self.a_min)
        a_max = copy.deepcopy(self.a_max)
        action = self.__class__(robot=robot, joint_ids=joints, f_min=a_min, f_max=a_max)
        memo[self] = action
        return action
