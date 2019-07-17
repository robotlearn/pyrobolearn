#!/usr/bin/env python
"""Define the various joint actions

This includes notably the joint positions, velocities, and force/torque actions.
"""

import copy
import numpy as np
from abc import ABCMeta

from pyrobolearn.actions.robot_actions.robot_actions import RobotAction, Robot


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
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
        """
        Initialize the joint position action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If None, it will apply the
                default maximum force values (read from the URDF).
        """
        super(JointPositionAction, self).__init__(robot, joint_ids)
        self.kp, self.kd, self.max_force = kp, kd, max_force

        # # check max force and take the one by default
        # if self.max_force is None:
        #     self.max_force = self.robot.get_joint_max_forces(self.joints)
        #     if np.allclose(self.max_force, 0):
        #         self.max_force = None

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


class JointPositionChangeAction(JointPositionAction):
    r"""Joint Position Change Action

    Set the joint positions using position control; this class expect to receive a change in the joint positions
    (i.e. instantaneous joint velocities). That is, the current joint positions are added to the given joint position
    changes. If none are provided, it will stay at the current configuration.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        """
        Initialize the joint position change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
        """
        super(JointPositionChangeAction, self).__init__(robot, joint_ids, kp=kp, kd=kd, max_force=max_force)
        self.data = np.zeros(len(self.joints))

    def _write(self, data):
        """apply the action data on the robot."""
        # add the original joint positions
        data += self.robot.get_joint_positions(self.joints)
        super(JointPositionChangeAction, self)._write(data)


class JointVelocityAction(JointAction):
    r"""Joint Velocity Action

    Set the joint velocities using velocity control.
    """

    def __init__(self, robot, joint_ids=None):
        """
        Initialize the joint velocity action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
        """
        super(JointVelocityAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_velocities(self.joints)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_joint_velocities(data, self.joints)


class JointVelocityChangeAction(JointAction):
    r"""Joint Velocity Change Action

    Set the joint velocities using velocity control;  this class expect to receive a change in the joint velocities
    (i.e. instantaneous joint accelerations). That is, the current joint velocities are added to the given joint
    velocity changes. If none are provided, it will keep the current joint velocities.
    """

    def __init__(self, robot, joint_ids=None):
        """
        Initialize the joint velocity change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
        """
        super(JointVelocityChangeAction, self).__init__(robot, joint_ids)
        self.data = np.zeros(len(self.joints))

    def _write(self, data):
        """apply the action data on the robot."""
        data += self.robot.get_joint_velocities(self.joints)
        super(JointVelocityChangeAction, self)._write(data)


class JointPositionAndVelocityAction(JointAction):
    r"""Joint position and velocity action

    Set the joint position using position control using PD control, where the constraint error to be minimized is
    given by: :math:`error = kp * (q^* - q) - kd * (\dot{q}^* - \dot{q})`.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        """
        Initialize the joint position and velocity action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
        """
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


class JointPositionAndVelocityChangeAction(JointPositionAndVelocityAction):
    r"""Joint position and velocity action

    Set the joint position using position control using PD control, where the constraint error to be minimized is
    given by: :math:`error = kp * (q^* - q) - kd * (\dot{q}^* - \dot{q})`.
    """

    def __init__(self, robot, joint_ids=None, kp=None, kd=None, max_force=None):
        """
        Initialize the joint position and velocity change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
        """
        super(JointPositionAndVelocityChangeAction, self).__init__(robot, joint_ids, kp=kp, kd=kd, max_force=max_force)
        self.data = np.zeros(2*len(self.joints))

    def _write(self, data):
        """apply the action data on the robot."""
        pos, vel = self.robot.get_joint_positions(self.joints), self.robot.get_joint_velocities(self.joints)
        data += np.concatenate((pos, vel))
        super(JointPositionAndVelocityChangeAction, self)._write(data)


# class JointPositionVelocityAccelerationAction(JointAction):
#     r"""Set the joint positions, velocities, and accelerations.
#
#     Set the joint positions, velocities, and accelerations by computing the necessary torques / forces using inverse
#     dynamics.
#     """
#     pass


class JointTorqueAction(JointAction):
    r"""Joint Torque/Force Action

    Set the joint force/torque using force/torque control.
    """

    def __init__(self, robot, joint_ids=None, f_min=-np.infty, f_max=np.infty):
        """
        Initialize the joint torque/force action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            f_min (float, np.array[N], None): minimum torques/forces.
            f_max (float, np.array[N], None): maximum torques/forces.
        """
        super(JointTorqueAction, self).__init__(robot, joint_ids)
        self.data = robot.get_joint_torques(self.joints)

        # check torque bounds
        if f_min is None or f_max is None:
            f = robot.get_joint_max_forces(joint_ids=self.joints)
            f_min = -f if f_min is None else f_min
            f_max = f if f_max is None else f_max
            if np.allclose(f_min, 0):
                f_min = -np.infty
            if np.allclose(f_max, 0):
                f_max = np.infty
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


# alias
JointForceAction = JointTorqueAction


class JointTorqueGravityCompensationAction(JointTorqueAction):
    r"""Joint torque action with gravity compensation enabled.

    This adds the given torques to the gravity compensation torques. That is, if a torque of 0 is provided, the robot
    will be in a gravity compensation mode.
    """
    def __init__(self, robot, joint_ids=None, f_min=-np.infty, f_max=np.infty):
        """
        Initialize the joint torque/force action with gravity compensation.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            f_min (float, np.array[N], None): minimum torques/forces.
            f_max (float, np.array[N], None): maximum torques/forces.
        """
        super(JointTorqueGravityCompensationAction, self).__init__(robot, joint_ids, f_min=f_min, f_max=f_max)
        self.q_indices = self.robot.get_q_indices(joint_ids=self.joints)
        self.data = np.zeros(len(self.joints))

    def _write(self, data):
        """apply the action data on the robot."""
        # add gravity compensation torques
        data += self.robot.get_gravity_compensation_torques(q_idx=self.q_indices)
        super(JointTorqueGravityCompensationAction, self)._write(data)


# alias
# JointForceGravityCompensationAction = JointTorqueGravityCompensationAction
JointTorqueChangeAction = JointTorqueGravityCompensationAction


class JointAccelerationAction(JointAction):
    r"""Joint Acceleration Action

    Set the joint accelerations using force/torque control. In order to produce the given joint accelerations,
    we use inverse dynamics which given the joint accelerations produce the corresponding joint forces/torques
    to be applied.
    """

    def __init__(self, robot, joint_ids=None, a_min=-np.infty, a_max=np.infty):
        """
        Initialize the joint acceleration action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            a_min (float, np.array[N], None): minimum accelerations.
            a_max (float, np.array[N], None): maximum accelerations.
        """
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
