#!/usr/bin/env python
"""Provide the Unmanned Underwater Vehicle (UUV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot


class UUVRobot(Robot):
    r"""Unmanned Underwater Vehicle Robot

    Vehicles/Robots that operate under water.
    """

    def __init__(self, simulator, urdf, position=None, orientation=None, fixed_base=False, scaling=1.):
        super(UUVRobot, self).__init__(simulator, urdf, position, orientation, fixed_base, scaling)
