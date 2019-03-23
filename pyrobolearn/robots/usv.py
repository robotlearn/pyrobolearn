#!/usr/bin/env python
"""Provide the Unmanned Surface Vehicle (USV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot


class USVRobot(Robot):
    r"""Unmanned Surface Vehicle Robot

    Vehicles/Robots that operate on the surface of water.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(USVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
