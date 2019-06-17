#!/usr/bin/env python
"""Provide the Unmanned Underwater Vehicle (UUV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class UUVRobot(Robot):
    r"""Unmanned Underwater Vehicle Robot

    Vehicles/Robots that operate under water.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scale=1.):
        super(UUVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
