#!/usr/bin/env python
"""Provide the Unmanned Surface Vehicle (USV) robot abstract classes.
"""

from pyrobolearn.robots.robot import Robot


class USVRobot(Robot):
    r"""Unmanned Surface Vehicle Robot

    Vehicles/Robots that operate on the surface of water.
    """

    def __init__(self, simulator, urdf_path, init_pos=(0, 0, 1.), init_orient=(0, 0, 0, 1), useFixedBase=False,
                 scaling=1.):
        super(USVRobot, self).__init__(simulator, urdf_path, init_pos, init_orient, useFixedBase, scaling)
