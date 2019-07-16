#!/usr/bin/env python
"""Define the various link / end-effector actions

This includes notably the link positions, velocities, and force/torque actions.
"""

import copy
from abc import ABCMeta
import numpy as np

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


class LinkAction(RobotAction):
    r"""Link Action
    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkAction, self).__init__(robot)

        # get the joints of the robot
        if link_ids is None:
            link_ids = robot.get_link_ids()
        self.links = link_ids

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(self.robot, self.links)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        robot = memo.get(self.robot, self.robot)  # copy.deepcopy(self.robot, memo)
        links = copy.deepcopy(self.links)
        action = self.__class__(robot, links)
        memo[self] = action
        return action


class LinkPositionAction(LinkAction):
    r"""Link world position action

    Set the world position using IK for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world position action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkPositionAction, self).__init__(robot, link_ids)
        self.data = self.robot.get_link_world_positions(link_ids=self.links, flatten=True)  # (N*3,)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(link_ids=self.links, positions=data.reshape(-1, 3))


class LinkPositionChangeAction(LinkPositionAction):
    r"""Link world position change action

    Set the world position using IK for the specified robot link(s). Instead of specifying directly the desired
    cartesian position(s), the amount of change in the current positions is provided.
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world position change action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkPositionChangeAction, self).__init__(robot, link_ids)
        self.data = np.zeros(len(self.links) * 3)

    def _write(self, data):
        """apply the action data on the robot."""
        data += self.robot.get_link_world_positions(link_ids=self.links)
        super(LinkPositionChangeAction, self)._write(data)


class LinkOrientationAction(LinkAction):
    r"""Link world orientation action

    Set the world orientation using IK for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world orientation action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkOrientationAction, self).__init__(robot, link_ids)
        self.data = self.robot.get_link_world_orientations(link_ids=self.links, flatten=True)  # (N*4,)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(link_ids=self.links, orientations=data.reshape(-1, 4))


class LinkOrientationChangeAction(LinkOrientationAction):
    r"""Link world orientation change action

    Set the world orientation using IK for the specified robot link(s). Instead of specifying directly the desired
    cartesian orientation(s), the amount of change in the current orientations is provided.

    Warnings: the difference in orientations should be provided as a change in roll-pitch-yaw angles (in radians), and
    not as quaternions.
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world orientation change action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkOrientationChangeAction, self).__init__(robot, link_ids)
        self.data = np.zeros(len(self.links) * 4)

    def _write(self, data):
        """apply the action data on the robot."""
        # get current orientations and convert them to RPY angles
        orientations = self.robot.get_link_world_orientations(link_ids=self.links).reshape(-1, 4)  # (N,4)
        orientations = get_rpy_from_quaternion(orientations)  # (N,3)

        # add change in orientations
        data = data.reshape(-1, 3)  # (N,3)
        data += orientations

        # convert them back to quaternions
        data = get_quaternion_from_rpy(data)  # (N,4)
        super(LinkOrientationChangeAction, self)._write(data)


class LinkPoseAction(LinkAction):
    r"""Link world pose action

    Set the world pose using IK for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world pose action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkPoseAction, self).__init__(robot, link_ids)
        self.data = self.robot.get_link_world_poses(link_ids=self.links, flatten=True)

    def _write(self, data):
        """apply the action data on the robot."""
        data = data.reshape(-1, 7)
        self.robot.set_link_positions(link_ids=self.links, positions=data[:, :3], orientations=data[:, 3:])


class LinkPoseChangeAction(LinkPoseAction):
    r"""Link world change pose action

    Set the world pose using IK for the specified robot link(s). Instead of specifying directly the desired
    cartesian pose(s), the amount of change in the current poses is provided.

    Warnings: the difference in orientations should be provided as a change in roll-pitch-yaw angles (in radians), and
    not as quaternions.
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world pose change action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkPoseChangeAction, self).__init__(robot, link_ids)
        self.data = np.zeros(len(self.links) * 6)

    def _write(self, data):
        """apply the action data on the robot."""
        # get current poses
        positions = self.robot.get_link_world_poses(link_ids=self.links, flatten=False).reshape(-1, 3)  # (N,3)
        orientations = self.robot.get_link_world_orientations(link_ids=self.links, flatten=False).reshape(-1, 4)  # (N,4)
        orientations = get_rpy_from_quaternion(orientations)  # (N,3)

        # add changes
        data = data.reshape(-1, 6)  # (N,6)
        data[:, :3] += positions
        data[:, 3:] += orientations

        # convert back orientations to quaternions
        data = np.hstack((data[:, :3], get_quaternion_from_rpy(data[:, 3:])))

        # write poses
        super(LinkPoseChangeAction, self)._write(data)


class LinkVelocityAction(LinkAction):
    r"""Link world velocity action

    Set the cartesian world velocity(ies) for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world velocity action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkVelocityAction, self).__init__(robot, link_ids)
        self.data = self.robot.get_link_world_velocities(link_ids=self.links)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_velocities(link_ids=self.links, positions=data.reshape(-1, 6))


class LinkVelocityChangeAction(LinkVelocityAction):
    r"""Link world velocity change action

    Set the cartesian world velocity(ies) for the specified robot link(s). Instead of specifying directly the desired
    cartesian velocity(ies), the amount of change in the current velocities is provided.
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link world velocity change action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkVelocityAction, self).__init__(robot, link_ids)
        self.data = np.zeros(len(self.links) * 6)

    def _write(self, data):
        """apply the action data on the robot."""
        data += self.robot.get_link_world_velocities(link_ids=self.links)
        super(LinkVelocityChangeAction, self)._write(data)


class LinkForceAction(LinkAction):
    r"""Link force action

    Set the cartesian force(s) for the specified robot link(s).
    """
    def __init__(self, robot, link_ids=None):
        """
        Initialize the link force action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkForceAction, self).__init__(robot, link_ids)

    def _write(self, data):
        """apply the action data on the robot."""
        # self.robot
        pass


########################
# End Effector Actions #
########################

class EndEffectorAction(LinkAction):

    def __init__(self, robot, end_effector_ids=None):
        if end_effector_ids is None:
            end_effector_ids = robot.get_end_effector_ids()
        super(EndEffectorAction, self).__init__(robot, end_effector_ids)


class EndEffectorPositionAction(EndEffectorAction):

    def __init__(self, robot, end_effector_ids=None):
        super(EndEffectorPositionAction, self).__init__(robot, end_effector_ids)


class EndEffectorVelocityAction(EndEffectorAction):

    def __init__(self, robot, end_effector_ids=None):
        super(EndEffectorVelocityAction, self).__init__(robot, end_effector_ids)


class EndEffectorForceAction(EndEffectorAction):

    def __init__(self, robot, end_effector_ids=None):
        super(EndEffectorForceAction, self).__init__(robot, end_effector_ids)
