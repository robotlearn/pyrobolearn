#!/usr/bin/env python
"""Define the various link / end-effector actions

This includes notably the link positions, velocities, and force/torque actions.
"""

import copy
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
    r"""Link position action

    Set the position using IK for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link position action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkPositionAction, self).__init__(robot, link_ids)

    def _write(self, data):
        """apply the action data on the robot."""
        self.robot.set_link_positions(self.links, data)


class LinkVelocityAction(LinkAction):
    r"""Link velocity action

    Set the cartesian velocity(ies) for the specified robot link(s).
    """

    def __init__(self, robot, link_ids=None):
        """
        Initialize the link position action.

        Args:
            robot (Robot): robot instance
            link_ids (int, int[N]): link id or list of link ids
        """
        super(LinkVelocityAction, self).__init__(robot, link_ids)

    def _write(self, data):
        """apply the action data on the robot."""
        # self.robot
        pass


class LinkForceAction(LinkAction):
    r"""Link force action

    Set the cartesian force(s) for the specified robot link(s).
    """
    def __init__(self, robot, link_ids=None):
        """
        Initialize the link position action.

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
