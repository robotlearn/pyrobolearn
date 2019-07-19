#!/usr/bin/env python
"""Define some robot terminal conditions for the environment.
"""

from abc import ABCMeta

from pyrobolearn.robots.robot import Robot
from pyrobolearn.terminal_conditions.body_conditions import BodyCondition


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotCondition(BodyCondition):
    r"""Robot Terminal Condition

    """
    __metaclass__ = ABCMeta

    def __init__(self, robot, bounds=(None, None), dim=None, out=False, stay=False, all=False):
        """
        Initialize the robot terminal condition.

        Args:
            robot (Robot): robot instance
            dim (None, int, int[3]): dimensions that we should consider when looking at the bounds. If None, it will
                consider all 3 dimensions. If one dimension is provided it will only check along that dimension. If
                a np.array of 0 and 1 is provided, it will consider the dimensions that are equal to 1. Thus, [1,0,1]
                means to consider the bounds along the x and z axes.
            out (bool): if True, we are outside the provided bounds. If False, we are inside the provided bounds.
            stay (bool): if True, it must stay in the bounds defined by in_bounds or out_bounds; if the state
                leave the bounds it results in a failure. if :attr:`stay` is False, it must get outside these bounds;
                if the state leaves the bounds, it results in a success.
            all (bool): this is only used if they are multiple dimensions. if True, all the dimensions of the state
                are checked if they are inside or outside the bounds depending on the other parameters. if False, any
                dimensions will be checked.
        """
        super(RobotCondition, self).__init__(robot, bounds=bounds, dim=dim, out=out, stay=stay, all=all)
        self.robot = robot

    @property
    def robot(self):
        """Return the robot instance."""
        return self._robot

    @robot.setter
    def robot(self, robot):
        """Set the robot instance."""
        if not isinstance(robot, Robot):
            raise TypeError("Expecting the given 'robot' to be an instance of `Robot`, instead got: "
                            "{}".format(type(robot)))
        self._robot = robot
