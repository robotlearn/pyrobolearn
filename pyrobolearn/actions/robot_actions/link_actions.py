#!/usr/bin/env python
"""Define the various link / end-effector actions

This includes notably the link positions, velocities, and force/torque actions.
"""

from abc import ABCMeta
from robot_actions import RobotAction


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
        Initialize the joint action.

        Args:
            robot (Robot): robot instance
            jointIds (int, int[N]): joint id or list of joint ids
        """
        super(LinkAction, self).__init__(robot)

        # get the joints of the robot
        if link_ids is None:
            link_ids = robot.getLinkIds()
        self.links = link_ids


class LinkPositionAction(LinkAction):
    r"""Link position action

    Set the link position(s) using IK.
    """
    def __init__(self, robot, link_ids=None):
        super(LinkPositionAction, self).__init__(robot, link_ids)

    def _write(self, data=None):
        if data is None:
            self.robot.setLinkPositions(self.links, self._data)
        else:
            self.robot.setLinkPositions(self.links, data)


########################
# End Effector Actions #
########################

class EndEffectorAction(LinkAction):

    def __init__(self, robot, end_effector_ids=None):
        if end_effector_ids is None:
            end_effector_ids = robot.getEndEffectorIds()
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
