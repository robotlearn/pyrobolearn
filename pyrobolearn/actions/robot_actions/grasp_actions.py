# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define grasping actions
"""

import copy
import numpy as np

from pyrobolearn.actions.robot_actions.robot_actions import RobotAction
from pyrobolearn.robots.gripper import Gripper


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class GraspAction(RobotAction):
    r"""Attach Action.

    This is allows to
    """

    def __init__(self, gripper):
        """
        Initialize the grasping action.

        Args:
            gripper (Gripper): a gripper instance.
        """
        super(GraspAction, self).__init__(robot=gripper)
        self.gripper = gripper

    @property
    def gripper(self):
        """Return the gripper instance."""
        return self._gripper

    @gripper.setter
    def gripper(self, gripper):
        if not isinstance(gripper, Gripper):
            raise TypeError("Expecting the given 'gripper' to be an instance of `Gripper`, but got instead: "
                            "{}".format(type(gripper)))
        self._gripper = gripper

    def _write(self, data):
        """
        Write the data.

        Args:
            data (int, np.ndarray): the continuous data representing the grasping strength.
        """
        if isinstance(data, np.ndarray):
            data = data[0]
        self.gripper.grasp(strength=data)

    def __copy__(self):
        """Return a shallow copy of the action. This can be overridden in the child class."""
        return self.__class__(gripper=self.gripper)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the action. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]
        gripper = copy.deepcopy(self.gripper)
        action = self.__class__(gripper=gripper)
        memo[self] = action
        return action
