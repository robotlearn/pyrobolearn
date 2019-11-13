#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the various link / end-effector actions

This includes notably the link positions, velocities, and force/torque actions.
"""

import copy
from abc import ABCMeta
import numpy as np
import gym

from pyrobolearn.actions.robot_actions.robot_actions import RobotAction
from pyrobolearn.utils.transformation import get_rpy_from_quaternion, get_quaternion_from_rpy


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class LinkAction(RobotAction):  # TODO: multiple links
    r"""Link Action
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it is the base.
            discrete_values (np.array[N, M], np.array[N], None): if provided, it represents the discrete values that
                the action can take. Note that the action is no more continuous and becomes discrete at that point.
                The first value will be the default value to be set if no data is provided.
        """
        super(LinkAction, self).__init__(robot)

        # get the link of the robot
        if link_id is None:
            link_id = -1
        self.link = int(link_id)

        # if discrete values, check the type and create the space
        if discrete_values is not None:
            if not isinstance(discrete_values, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'discrete_values' to be a list/tuple/np.array of float/int, but "
                                "instead got: {}".format(type(discrete_values)))
            discrete_values = np.asarray(discrete_values)
            self._space = gym.spaces.Discrete(len(discrete_values))
        self.discrete_values = discrete_values

        # set the data in the case it is discrete
        if self.discrete_values is not None:
            self.data = np.zeros(1, dtype=np.int)  # set the data to be the first index

    def _check_discrete_values(self, dim, last_dim):
        """Check that the discrete values have the correct dimensions / shape."""
        # check discrete values
        if self.discrete_values is not None:
            if not (len(self.discrete_values.shape) == dim and self.discrete_values.shape[-1] == last_dim):
                raise ValueError("Expecting the discrete values to have a dimension of {} and the last value of the "
                                 "shape to be equal to {}, but instead got respectively: "
                                 "{}, {}".format(dim, last_dim, len(self.discrete_values.shape),
                                                 self.discrete_values.shape[-1]))

    def _write(self, data):
        """
        Write the data.

        Args:
            data (int, np.ndarray): the data can be discrete or continuous.
        """
        # if the action is discrete, then the data should be an index, or an array of values from which takes the max
        if self.is_discrete():
            if isinstance(data, np.ndarray):
                data = np.argmax(data.reshape(-1))
            data = self.discrete_values[data]
        self._write_continuous(data)

    def _write_continuous(self, data):
        """
        Write the given continuous data. Child method that has to be implement in the child classes.

        Args:
            data (np.ndarray): continuous data to be written.
        """
        raise NotImplementedError

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(self.robot, self.link)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        links = copy.deepcopy(self.link)
        action = self.__class__(robot, links)
        memo[self] = action
        return action


class LinkPositionAction(LinkAction):  # TODO: multiple links
    r"""Link world position action

    Set the world position using IK for the specified robot link.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world position action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N,3], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. Also, note
                that this parameter makes probably  more sense with `LinkPositionChangeAction` instead. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkPositionAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=3)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = self.robot.get_link_world_positions(link_ids=self.link, flatten=True)  # (3,)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(link_ids=self.link, positions=data)


class LinkPositionChangeAction(LinkAction):  # TODO: multiple links
    r"""Link world position change action

    Set the world position using IK for the specified robot link. Instead of specifying directly the desired
    cartesian position(s), the amount of change in the current positions is provided.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world position change action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N,3], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkPositionChangeAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=3)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        data += self.robot.get_link_world_positions(link_ids=self.link, flatten=True)  # (3,)
        self.robot.set_link_positions(link_ids=self.link, positions=data)


class LinkOrientationAction(LinkAction):  # TODO: multiple links
    r"""Link world orientation action

    Set the world orientation using IK for the specified robot link.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world orientation action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N,4], None): if provided, it represents the discrete values that the action can
                take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkOrientationAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=4)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = self.robot.get_link_world_orientations(link_ids=self.link, flatten=True)  # (4,)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(link_ids=self.link, orientations=data)


class LinkOrientationChangeAction(LinkAction):  # TODO: multiple links
    r"""Link world orientation change action

    Set the world orientation using IK for the specified robot link. Instead of specifying directly the desired
    cartesian orientation(s), the amount of change in the current orientations is provided.

    Warnings: the difference in orientations should be provided as a change in roll-pitch-yaw angles (in radians), and
    not as quaternions.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world orientation change action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N,3], None): if provided, it represents the discrete values that the orientation
                change action can take. The orientations are represented as Roll-Pitch-Yaw angles. Note that the
                action is no more continuous and becomes discrete at that point. The first value will be the default
                value to be set if no data is provided.
        """
        super(LinkOrientationChangeAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=3)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        # get current orientations and convert them to RPY angles
        orientation = self.robot.get_link_world_orientations(link_ids=self.link)  # (4,)
        orientation = get_rpy_from_quaternion(orientation)  # (3,)

        # add change in orientations
        data += orientation  # (3,)

        # convert them back to quaternions
        data = get_quaternion_from_rpy(data)  # (4,)
        self.robot.set_link_positions(link_ids=self.link, orientations=data)


class LinkPoseAction(LinkAction):  # TODO: multiple link
    r"""Link world pose action

    Set the world pose using IK for the specified robot link.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world pose action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N,7], None): if provided, it represents the discrete values that the action can
                take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkPoseAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=7)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = self.robot.get_link_world_poses(link_ids=self.link, flatten=True)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(link_ids=self.link, positions=data[:3], orientations=data[3:])


class LinkPoseChangeAction(LinkAction):  # TODO: multiple link
    r"""Link world change pose action

    Set the world pose using IK for the specified robot link. Instead of specifying directly the desired
    cartesian pose(s), the amount of change in the current poses is provided.

    Warnings: the difference in orientations should be provided as a change in roll-pitch-yaw angles (in radians),
    and not as quaternions.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world pose change action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N, 6], None): if provided, it represents the discrete values that the link pose
                change action can take. Note that the orientation part is represented as Roll-Pitch-Yaw angles.
                Note that the action is no more continuous and becomes discrete at that point. The first value will
                be the default value to be set if no data is provided.
        """
        super(LinkPoseChangeAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=6)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = np.zeros(6)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        # get current pose
        position = self.robot.get_link_world_poses(link_ids=self.link, flatten=False)  # (3,)
        orientation = self.robot.get_link_world_orientations(link_ids=self.link, flatten=False)  # (4,)
        orientation = get_rpy_from_quaternion(orientation)  # (3,)

        # add changes
        data[:3] += position        # (3,)
        data[3:] += orientation     # (3,)

        # convert back orientations to quaternions
        data = np.concatenate((data[:3], get_quaternion_from_rpy(data[3:])))  # (7,)

        # write poses
        self.robot.set_link_positions(link_ids=self.link, positions=data[:3], orientations=data[3:])


class LinkVelocityAction(LinkAction):  # TODO: multiple links
    r"""Link world velocity action

    Set the cartesian world velocity for the specified robot link.
    """

    def __init__(self, robot, link_id=-1, discrete_values=None):
        """
        Initialize the link world velocity action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N, 6], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkVelocityAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=6)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = self.robot.get_link_world_velocities(link_ids=self.link)  # (6,)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_velocities(link_ids=self.link, positions=data)


class LinkVelocityChangeAction(LinkAction):  # TODO: multiple link
    r"""Link world velocity change action

    Set the cartesian world velocity for the specified robot link. Instead of specifying directly the desired
    cartesian velocity, the amount of change in the current velocities is provided.
    """

    def __init__(self, robot, link_id=None, discrete_values=None):
        """
        Initialize the link world velocity change action.

        Args:
            robot (Robot): robot instance.
            link_id (int): link id. If -1, it represents the base.
            discrete_values (np.array[N, 6], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkVelocityChangeAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=6)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = np.zeros(6)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        data += self.robot.get_link_world_velocities(link_ids=self.link)
        self.robot.set_link_velocities(link_ids=self.link, positions=data)


class LinkForceAction(LinkAction):  # TODO: multiple links
    r"""Link force action

    Set the robot joint torques in order to perform a desired cartesian force(s) with the specified robot link on
    the environment. The final joint torques that are applied are:

    .. math:: \tau = N(q,\dot{q}) + J(q)^\top f

    where :math:`N(q,\dot{q})` contains the coriolis, centrifugal, and gravity effects, and :math:`f` is the cartesian
    force that we wish to apply on the environment.
    """

    def __init__(self, robot, link_id, discrete_values=None):  # link_ids=None
        """
        Initialize the link force action.

        Args:
            robot (Robot): robot instance
            link_id (int): id of the link that has to perform the desired force.
            discrete_values (np.array[N, 3], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkForceAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=3)

        # set the original data
        if self.is_continuous():  # continuous action  # TODO: check if we can sense the forces
            self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        jacobian = self.robot.get_jacobian(link_id=self.link)[:3]  # (3,N)
        tau = self.robot.get_coriolis_and_gravity_compensation_torques()  # (N,)
        tau += jacobian.T.dot(data)  # (N,)
        self.robot.set_joint_torques(tau)


class LinkTorqueAction(LinkAction):  # TODO: multiple links
    r"""Link torque action

    Set the robot joint torques in order to perform a desired cartesian torque with the specified link on the
    environment. The final joint torques that are applied are:

    .. math:: \tau = N(q,\dot{q}) + J(q)^\top f

    where :math:`N(q,\dot{q})` contains the coriolis, centrifugal, and gravity effects, and :math:`f` is the cartesian
    torque that we wish to apply on the environment.
    """

    def __init__(self, robot, link_id, discrete_values=None):  # link_ids=None
        """
        Initialize the link torque action.

        Args:
            robot (Robot): robot instance.
            link_id (int): id of the link that has to perform the desired torque.
            discrete_values (np.array[N, 3], None): if provided, it represents the discrete values that the action
                can take. Note that the action is no more continuous and becomes discrete at that point. The first
                value will be the default value to be set if no data is provided.
        """
        super(LinkTorqueAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=3)

        # set the original data
        if self.is_continuous():  # continuous action  # TODO: check if we can sense the torques
            self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        jacobian = self.robot.get_jacobian(link_id=self.link)[3:]  # (3,N)
        tau = self.robot.get_coriolis_and_gravity_compensation_torques()  # (N,)
        tau += jacobian.T.dot(data)  # (N,)
        self.robot.set_joint_torques(tau)


class LinkWrenchAction(LinkAction):  # TODO: multiple links
    r"""Link wrench action

    Set the robot joint torques in order to perform a desired cartesian wrench (concatenation of the cartesian force
    and torque) with the specified link on the environment.  The final joint torques that are applied are:

    .. math:: \tau = N(q,\dot{q}) + J(q)^\top f

    where :math:`N(q,\dot{q})` contains the coriolis, centrifugal, and gravity effects, and :math:`f` is the cartesian
    wrench that we wish to apply on the environment.
    """

    def __init__(self, robot, link_id, discrete_values=None):  # link_ids=None
        """
        Initialize the link wrench action.

        Args:
            robot (Robot): robot instance.
            link_id (int): id of the link that has to perform the desired wrench.
            discrete_values (np.array[N,6], None): if provided, it represents the discrete values that the
                action can take. Note that the action is no more continuous and becomes discrete at that point.
                The first value will be the default value to be set if no data is provided.
        """
        super(LinkWrenchAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check discrete values
        self._check_discrete_values(dim=2, last_dim=6)

        # set the original data
        if self.is_continuous():  # continuous action
            self.data = np.zeros(6)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        jacobian = self.robot.get_jacobian(link_id=self.link)  # (6,N)
        tau = self.robot.get_coriolis_and_gravity_compensation_torques()  # (N,)
        tau += jacobian.T.dot(data)  # (N,)
        self.robot.set_joint_torques(tau)


class ApplyForceAction(LinkAction):  # TODO: multiple links
    r"""Apply Force Action

    This action allows you to apply a Cartesian force on a specific link. In the simulator, it just applies the force
    on the specified link at the specified position. On the real platform, it projects the Cartesian force to joint
    torques and apply them on the robot.
    """

    def __init__(self, robot, link_id=-1, local_position=None, axis=None, discrete_values=None):  # link_ids=None
        """
        Initialize the apply force action.

        Args:
            robot (Robot): robot instance.
            link_id (int): id of the link on which to apply the force. If -1, it is the base.
            local_position (np.array[3], list of 3 float, None): local position on the link to apply the force on.
                If None, the force will be applied on the CoM of the link.
            axis (np.array[3], None): axis on which to apply the force. If provided, it will create an action that
                only represents the magnitude of the force.
            discrete_values (np.array[N,3], np.array[N], None): if provided, it represents the forces in a discrete
                manner by using the provided force values. Note that the action is no more continuous and becomes
                discrete at that point. The first value will be the default value to be set if no data is provided.
        """
        super(ApplyForceAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check local position
        if local_position is not None:
            if not isinstance(local_position, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'local_position' to be a list/tuple/np.array of 3 float, or None, "
                                "but instead got: {}".format(type(local_position)))
            if len(local_position) != 3:
                raise ValueError("Expecting the given 'local_position' to be a list/tuple/np.array of 3 float, but "
                                 "instead got a length of: {}".format(len(local_position)))
        self.local_position = local_position

        # check axis
        if axis is not None:
            if not isinstance(axis, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'axis' to be a list/tuple/np.array, but instead got: "
                                "{}".format(type(axis)))
            axis = np.asarray(axis)
            if axis.size != 3:
                raise ValueError("Expecting the given 'axis' to be list/tuple/np.array of 3 float, but instead got: "
                                 "{}".format(axis.size))
        self.axis = axis

        # check discrete values
        if self.discrete_values is not None:
            # if an axis is not defined the discrete values must have a shape of (N,2)
            if self.axis is None:
                if not (len(self.discrete_values.shape) == 2 and self.discrete_values.shape[1] == 3):
                    raise ValueError("Expecting the discrete values to have a shape of (N,3), but instead got: "
                                     "{}".format(self.discrete_values.shape))
            else:  # if an axis is defined, the discrete values must have a shape of (N,)
                if len(self.discrete_values.shape) > 1:
                    raise ValueError("Expecting the discrete values to have a shape of (N,), but instead got: "
                                     "{}".format(self.discrete_values.shape))

        # set the original data
        if self.is_continuous():  # continuous action
            if self.axis is not None:  # if an axis is defined, then set the initial data to be zero
                self.data = np.zeros(1)
            else:  # if no axis is defined, set the initial data to be a 3D zero vector
                self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        if self.axis is not None:
            data = data * self.axis
        self.robot.apply_external_force(force=data, link_id=self.link, position=self.local_position)


class ApplyTorqueAction(LinkAction):  # TODO: multiple links
    r"""Apply Torque Action

    This action allows you to apply a Cartesian torque on a specific link. In the simulator, it just applies the torque
    on the specified link at the specified position. On the real platform, it projects the Cartesian torque to joint
    torques and apply them on the robot.
    """

    def __init__(self, robot, link_id=-1, axis=None, discrete_values=None):  # link_ids=None
        """
        Initialize the apply torque action.

        Args:
            robot (Robot): robot instance.
            link_id (int): id of the link on which to apply the torque. If -1, it is the base.
            axis (np.array[3], None): axis around which to apply the torque. If provided, it will create an action that
                only represents the magnitude of the torque.
            discrete_values (np.array[N,3], np.array[N], None): if provided, it represents the torques in a discrete
                manner by using the provided torque values. Note that the action is no more continuous and becomes
                discrete at that point. The first value will be the default value to be set if no data is provided.
        """
        super(ApplyTorqueAction, self).__init__(robot, link_id, discrete_values=discrete_values)

        # check axis
        if axis is not None:
            if not isinstance(axis, (list, tuple, np.ndarray)):
                raise TypeError("Expecting the given 'axis' to be a list/tuple/np.array, but instead got: "
                                "{}".format(type(axis)))
            axis = np.asarray(axis)
            if axis.size != 3:
                raise ValueError("Expecting the given 'axis' to be list/tuple/np.array of 3 float, but instead got: "
                                 "{}".format(axis.size))
        self.axis = axis

        # check discrete values
        if self.discrete_values is not None:
            # if an axis is not defined the discrete values must have a shape of (N,2)
            if self.axis is None:
                if not (len(self.discrete_values.shape) == 2 and self.discrete_values.shape[1] == 3):
                    raise ValueError("Expecting the discrete values to have a shape of (N,3), but instead got: "
                                     "{}".format(self.discrete_values.shape))
            else:  # if an axis is defined, the discrete values must have a shape of (N,)
                if len(self.discrete_values.shape) > 1:
                    raise ValueError("Expecting the discrete values to have a shape of (N,), but instead got: "
                                     "{}".format(self.discrete_values.shape))

        # set the original data
        if self.is_continuous():  # continuous action
            if self.axis is not None:
                # if an axis is defined, then set the initial data to be zero
                self.data = np.zeros(1)
            else:
                self.data = np.zeros(3)

    def _write_continuous(self, data):
        """apply the action data on the robot."""
        if self.axis is not None:
            data = data * self.axis
        self.robot.apply_external_torque(torque=data, link_id=self.link)


# class ApplyWrenchAction(LinkAction):  # TODO: multiple links
#     r"""Apply wrench action
#
#     This action allows you to apply a Cartesian wrench (concatenation of the force and torque) on a specific link.
#     In the simulator, it just applies the wrench on the specified link at the specified position. On the real
#     platform, it projects the Cartesian wrench to joint torques and apply them on the robot.
#     """
#
#     def __init__(self, robot, link_id, local_position=None, axis=None, discrete_values=None):  # link_ids=None
#         """
#         Initialize the apply wrench action.
#
#         Args:
#             robot (Robot): robot instance.
#             link_id (int): id of the link on which to apply the force. If -1, it is the base.
#             local_position (np.array[3], list of 3 float, None): local position on the link to apply the force on.
#                 If None, the force will be applied on the CoM of the link.
#             axis (np.array[3], None): axis on which to apply the force. If provided, it will create an action that
#                 only represents the magnitude of the force.
#             discrete_values (np.array[M]): if provided, it represents the forces in a discrete manner by using the
#                 provided force value. Note that the action is no more continuous and becomes discrete at that point.
#         """
#         super(ApplyWrenchAction, self).__init__(robot, link_id, discrete_values=discrete_values)
#
#     def _write(self, data):
#         """apply the action data on the robot."""
#         pass


########################
# End Effector Actions #
########################

# class EndEffectorAction(LinkAction):
#
#     def __init__(self, robot, end_effector_ids=None):
#         if end_effector_ids is None:
#             end_effector_ids = robot.get_end_effector_ids()
#         super(EndEffectorAction, self).__init__(robot, end_effector_ids)
#
#
# class EndEffectorPositionAction(EndEffectorAction):
#
#     def __init__(self, robot, end_effector_ids=None):
#         super(EndEffectorPositionAction, self).__init__(robot, end_effector_ids)
#
#
# class EndEffectorVelocityAction(EndEffectorAction):
#
#     def __init__(self, robot, end_effector_ids=None):
#         super(EndEffectorVelocityAction, self).__init__(robot, end_effector_ids)
#
#
# class EndEffectorForceAction(EndEffectorAction):
#
#     def __init__(self, robot, end_effector_ids=None):
#         super(EndEffectorForceAction, self).__init__(robot, end_effector_ids)
