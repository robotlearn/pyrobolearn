#!/usr/bin/env python
"""Define the various joint actions

This includes notably the joint positions, velocities, and force/torque actions.
"""

import copy
import numpy as np
from abc import ABCMeta
import gym

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

    def __init__(self, robot, joint_ids=None, discrete_values=None):
        """
        Initialize the joint action.

        Args:
            robot (Robot): robot instance
            joint_ids (int, int[N]): joint id or list of joint ids
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointAction, self).__init__(robot)

        # get the joints of the robot
        if joint_ids is None:
            joint_ids = robot.get_joint_ids()
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        self.joints = joint_ids

        # if discrete values, check the type and set the space
        if discrete_values is not None:

            # check the type
            if not isinstance(discrete_values, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'discrete_values' to be a list/tuple/np.array of float/int, but "
                                "instead got: {}".format(type(discrete_values)))

            if len(discrete_values) == 0:
                raise ValueError("Expecting at least one list of discrete values")
            if not isinstance(discrete_values[0], (list, tuple, np.ndarray)):
                discrete_values = [discrete_values]

            # check that the number of list of discrete values match the number of joints
            if len(discrete_values) != len(self.joints):
                raise ValueError("The number of discrete value sets (={}) does not match the number of joints "
                                 "(={})".format(len(discrete_values), len(self.joints)))

            # check the type and shape of each discrete value set, and convert it to numpy arrays
            for i, value in enumerate(discrete_values):
                if not isinstance(value, (list, tuple, np.ndarray)):
                    raise TypeError("Expecting each discrete value set to be a list/tuple/np.ndarray, instead got: "
                                    "{} at index {}".format(type(value), i))
                discrete_values[i] = np.asarray(value)
                if len(discrete_values.shape) != 1:
                    raise ValueError("Expecting each discrete value set to be a 1D array, instead got a shape of: "
                                     "{}".format(discrete_values.shape))

        # set the discrete values
        self.discrete_values = discrete_values

        # set the data and the space in the case of discrete values
        if self.discrete_values is not None:
            # set the space
            if len(self.discrete_values) == 1:
                self._space = gym.spaces.Discrete(len(self.discrete_values))
            else:
                self._space = gym.spaces.MultiDiscrete([len(value) for value in self.discrete_values])

            # set the data
            if isinstance(self._space, gym.spaces.Discrete):
                self.data = 0
            else:
                self.data = np.zeros(len(self._space.nvec))

    # @property
    # def size(self):
    #     return len(self.joints)

    def _check_continuous_bounds(self, bounds):
        """Check the given continuous bounds."""
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("Expecting the given bounds to be a tuple/list/np.ndarray of float, instead got: "
                            "{}".format(type(bounds)))
        if len(bounds) != 2:
            raise ValueError("Expecting the bounds to be of length 2 (i.e. lower and upper bounds), instead got a "
                             "length of {}".format(len(bounds)))
        if bounds[0] is not None and bounds[1] is not None:
            bounds = np.asarray(bounds).reshape(2, -1)
            if len(self.joints) != bounds.shape[1]:
                if bounds.shape[1] == 1:
                    bounds = np.array([bounds[0, 0] * np.ones(len(self.joints)),
                                       bounds[1, 0] * np.ones(len(self.joints))])
                else:
                    raise ValueError("Expecting the number of bounds to match up with the number of joints")
        else:
            bounds = tuple(bounds)
        return bounds

    # def _write(self, data):
    #     """
    #     Write the data.
    #
    #     Args:
    #         data (int, np.ndarray): the data can be discrete or continuous.
    #     """
    #     # if the action is discrete, then the data should be an index, or an array of values from which takes the max
    #     if self.is_discrete():
    #         if isinstance(data, np.ndarray):
    #             data = np.argmax(data.reshape(-1))
    #         data = self.discrete_values[data]
    #     self._write_continuous(data)
    #
    # def _write_continuous(self, data):
    #     """
    #     Write the given continuous data. Child method that has to be implement in the child classes.
    #
    #     Args:
    #         data (np.ndarray): continuous data to be written.
    #     """
    #     raise NotImplementedError

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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), kp=None, kd=None, max_force=None,
                 discrete_values=None):
        """
        Initialize the joint position action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): lower and upper bound in the case of continuous action.
                If None it will use the default joint position limits.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If None, it will apply the
                default maximum force values (read from the URDF).
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointPositionAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)
        self.kp, self.kd, self.max_force = kp, kd, max_force

        # check max force and take the one by default
        if self.max_force is None:
            self.max_force = self.robot.get_joint_max_forces(self.joints)
            if np.allclose(self.max_force, 0):
                self.max_force = None

        # set data and space if continuous
        if self.discrete_values is None:
            self.data = self.robot.get_joint_positions(self.joints)
            bounds = self._check_continuous_bounds(bounds)
            if bounds == (None, None):
                bounds = self.robot.get_joint_limits(self.joints)
            self._space = gym.spaces.Box(low=bounds[:, 0], high=bounds[:, 1])

    def bounds(self):
        """Return the joint limits."""
        return self.robot.get_joint_limits(self.joints)

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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), kp=None, kd=None, max_force=None,
                 discrete_values=None):
        """
        Initialize the joint position change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): lower and upper bound in the case of continuous action.
                If None it will use the default joint position limits.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointPositionChangeAction, self).__init__(robot, joint_ids, bounds=bounds, kp=kp, kd=kd,
                                                        max_force=max_force, discrete_values=discrete_values)

        # set data if continuous
        if self.discrete_values is None:
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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), discrete_values=None):
        """
        Initialize the joint velocity action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): lower and upper bound in the case of continuous action.
                If None it will use the default joint position limits.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointVelocityAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)

        # set data and space if continuous
        if self.discrete_values is None:
            self.data = robot.get_joint_velocities(self.joints)
            bounds = self._check_continuous_bounds(bounds)
            if bounds == (None, None):
                bounds = self.robot.get_joint_max_velocities(self.joints)
                if np.allclose(bounds, 0):
                    bounds = np.array([-np.infty * np.ones(len(self.joints)),
                                       np.infty * np.ones(len(self.joints))])
            self._space = gym.spaces.Box(low=bounds[:, 0], high=bounds[:, 1])

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_joint_velocities(data, self.joints)


class JointVelocityChangeAction(JointAction):
    r"""Joint Velocity Change Action

    Set the joint velocities using velocity control;  this class expect to receive a change in the joint velocities
    (i.e. instantaneous joint accelerations). That is, the current joint velocities are added to the given joint
    velocity changes. If none are provided, it will keep the current joint velocities.
    """

    def __init__(self, robot, joint_ids=None, bounds=(None, None), discrete_values=None):
        """
        Initialize the joint velocity change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): lower and upper bound in the case of continuous action.
                If None it will use the default joint position limits.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointVelocityChangeAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)

        # set data if continuous
        if self.discrete_values is None:
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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), kp=None, kd=None, max_force=None,
                 discrete_values=None):
        """
        Initialize the joint position and velocity action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointPositionAndVelocityAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)
        self.kp, self.kd, self.max_force = kp, kd, max_force

        # set data if continuous
        if self.discrete_values is None:
            pos, vel = robot.get_joint_positions(self.joints), robot.get_joint_velocities(self.joints)
            self.data = np.concatenate((pos, vel))
        self.idx = len(self.joints)

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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), kp=None, kd=None, max_force=None,
                 discrete_values=None):
        """
        Initialize the joint position and velocity change action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id(s). If None, it will take all the actuated joints.
            kp (float, np.array[N], None): position gain(s)
            kd (float, np.array[N], None): velocity gain(s)
            max_force (float, np.array[N], None, bool): maximum motor torques / forces. If True, it will apply the
                default maximum force values.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointPositionAndVelocityChangeAction, self).__init__(robot, joint_ids, kp=kp, kd=kd, max_force=max_force,
                                                                   discrete_values=discrete_values)
        # set data if continuous
        if self.discrete_values is None:
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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), discrete_values=None):
        """
        Initialize the joint torque/force action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): minimum and maximum torques/forces respectively. If None,
                it will check the minimum/maximum torques/forces allowed. If it doesn't find them, it will set them
                to -np.infty and np.infty.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointTorqueAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)

        # check torque bounds
        f_min, f_max = self._check_continuous_bounds(bounds)
        if f_min is None or f_max is None:
            f = robot.get_joint_max_forces(joint_ids=self.joints)
            f_min = -f if f_min is None else f_min
            f_max = f if f_max is None else f_max
            if np.allclose(f_min, 0):
                f_min = -np.infty * np.ones(len(self.joints))
            if np.allclose(f_max, 0):
                f_max = np.infty * np.ones(len(self.joints))
        self.f_min = f_min
        self.f_max = f_max

        # set data and space if continuous
        if self.discrete_values is None:
            self.data = robot.get_joint_torques(self.joints)
            self._space = gym.spaces.Box(low=self.f_min, high=self.f_max)

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
    def __init__(self, robot, joint_ids=None, bounds=(None, None), discrete_values=None):
        """
        Initialize the joint torque/force action with gravity compensation.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): minimum and maximum torques/forces respectively. If None,
                it will check the minimum/maximum torques/forces allowed. If it doesn't find them, it will set them
                to -np.infty and np.infty.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointTorqueGravityCompensationAction, self).__init__(robot, joint_ids, bounds=bounds,
                                                                   discrete_values=discrete_values)
        self.q_indices = self.robot.get_q_indices(joint_ids=self.joints)

        # set data if continuous
        if self.discrete_values is None:
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

    def __init__(self, robot, joint_ids=None, bounds=(None, None), discrete_values=None):
        """
        Initialize the joint acceleration action.

        Args:
            robot (Robot): robot instance.
            joint_ids (int, list of int, None): joint id, or list of joint ids. If None, get all the actuated joints.
            bounds (tuple of 2 float / np.array[N] / None): minimum and maximum accelerations. If None, it will use
                the default joint acceleration limits. If it still doesn't find them, it will set -np.infty and
                np.infty.
            discrete_values (np.array[M], np.array[N,M], list of np.array[M], None): discrete values for each joint.
                Note that by specifying this, the joint action is no more continuous but becomes discrete. By default,
                the first value along the first axis / dimension are the values by default that are set if no data
                is provided.
        """
        super(JointAccelerationAction, self).__init__(robot, joint_ids, discrete_values=discrete_values)

        # TODO
        self.a_min = bounds[0]
        self.a_max = bounds[1]

        # set data if continuous
        if self.discrete_values is None:
            self.data = robot.get_joint_accelerations(self.joints)

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
