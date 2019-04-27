#!/usr/bin/env python
"""Provide the Unmanned Surface Vehicle (USV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class USVRobot(Robot):
    r"""Unmanned Surface Vehicle Robot

    Vehicles/Robots that operate on the surface of water.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(USVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
